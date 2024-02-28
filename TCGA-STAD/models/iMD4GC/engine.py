import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc

from .util import ModelEma

import torch.optim
import torch.nn.parallel


class Engine(object):
    def __init__(self, args, results_dir, fold, alpha, beta):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        print("[hyperparameter] alpha: {}, beta: {}".format(alpha, beta))
        self.alpha = alpha
        self.beta = beta
        # tensorboard
        if args.log_data and (not self.args.evaluate):
            from tensorboardX import SummaryWriter

            writer_dir = os.path.join(results_dir, "fold_" + str(fold))
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            self.writer = None
        self.best_scores = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        # self.scaler = GradScaler()
        if self.args.resume is not None:
            fielname = None
            if os.path.exists(os.path.join(self.args.resume, "fold_" + str(self.fold))):
                files = os.listdir(os.path.join(self.args.resume, "fold_" + str(self.fold)))
                for file in files:
                    if file.endswith(".pth.tar"):
                        fielname = file
                        break
            if fielname is not None:
                filename = os.path.join(self.args.resume, "fold_" + str(self.fold), fielname)
                print("=> loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                self.best_scores = checkpoint["best_score"]
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint (score: {})".format(checkpoint["best_score"]))
            else:
                print("=> no checkpoint found at '{}'".format(os.path.join(self.args.resume, "fold_" + str(self.fold))))

        if self.args.evaluate:
            print("=> perform evaluation on training set")
            _, survival_train, _ = self.validate(train_loader, model, criterion)
            print("=> perform evaluation on validation set")
            score, survival_test, estimate = self.validate(val_loader, model, criterion)
            survival_train = np.array(survival_train, dtype=[("death", np.dtype("bool")), ("OS", np.dtype("float64"))])
            survival_test = np.array(survival_test, dtype=[("death", np.dtype("bool")), ("OS", np.dtype("float64"))])
            # times = np.percentile(survival_test["OS"], np.linspace(20, 80, 4))
            times = np.array([7.06666667, 11.83333333, 18.56666667])
            auc, mean_auc = cumulative_dynamic_auc(survival_train, survival_test, estimate, times)
            return score, mean_auc

        ema_model = ModelEma(model, device="cuda" if torch.cuda.is_available() else "cpu")
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, ema_model, criterion, optimizer)
            # evaluate on validation set
            scores, _, _ = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = scores > self.best_scores
            if is_best:
                self.best_scores = scores
                self.best_epoch = self.epoch
                self.save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "best_score": self.best_scores})
            print(" *** best score={:.4f} at epoch {}".format(self.best_scores, self.best_epoch))
            scheduler.step()
            print(">>>")
            print(">>>")
        return self.best_scores, self.best_epoch

    def train(self, data_loader, model, ema_model, criterion, optimizer):
        # teacher model and student model
        model.train()
        ema_model.eval()
        # criterions
        criterion_cls = criterion[0]
        criterion_div = criterion[1]
        criterion_kd = criterion[2]

        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc="Train Epoch {}".format(self.epoch))
        for batch_idx, (data_ID, data_OS, data_Censorship, data_Label, data_WSI, data_Omics, data_Clinical) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                clinical = data_Clinical.type(torch.FloatTensor).cuda()
                label = data_Label.type(torch.LongTensor).cuda()
                censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # outputs of teacher model
            with torch.no_grad():
                hazards_t, _, feat_t = ema_model.module(
                    x_WSI=data_WSI,
                    x_Omics=data_Omics,
                    x_age=clinical[:, 0],
                    x_gender=clinical[:, 1],
                    x_pathology=clinical[:, 2],
                    x_t_stage=clinical[:, 3],
                    x_n_stage=clinical[:, 4],
                    x_m_stage=clinical[:, 5],
                    x_lymph=clinical[:, 6],
                    x_race=clinical[:, 7],
                )
            # outputs of student model
            hazards_s, S_s, feat_s = model(
                x_WSI=data_WSI,
                x_Omics=data_Omics,
                x_age=clinical[:, 0],
                x_gender=clinical[:, 1],
                x_pathology=clinical[:, 2],
                x_t_stage=clinical[:, 3],
                x_n_stage=clinical[:, 4],
                x_m_stage=clinical[:, 5],
                x_lymph=clinical[:, 6],
                x_race=clinical[:, 7],
            )
            loss_cls = criterion_cls(hazards=hazards_s, S=S_s, Y=label, c=censorship)
            loss_div = criterion_div(hazards_s, hazards_t)
            loss_kd = criterion_kd(feat_s, feat_t)
            # WSI
            if data_WSI.shape[1] > 1:
                hazards_WSI, S_WSI, feat_WSI = model(
                    x_WSI=data_WSI,
                    x_Omics=torch.zeros((1, 1)),
                    x_age=clinical[:, 0],
                    x_gender=clinical[:, 1],
                    x_pathology=clinical[:, 2],
                    x_t_stage=clinical[:, 3],
                    x_n_stage=clinical[:, 4],
                    x_m_stage=clinical[:, 5],
                    x_lymph=clinical[:, 6],
                    x_race=clinical[:, 7],
                )
                loss_cls = loss_cls + criterion_cls(hazards=hazards_WSI, S=S_WSI, Y=label, c=censorship)
                loss_div = loss_div + criterion_div(hazards_WSI, hazards_t)
                loss_kd = loss_kd + criterion_kd(feat_WSI, feat_t)
            # Omics
            if len(data_Omics) > 1:
                hazards_Omics, S_Omics, feat_Omics = model(
                    x_WSI=torch.zeros((1, 1)),
                    x_Omics=data_Omics,
                    x_age=clinical[:, 0],
                    x_gender=clinical[:, 1],
                    x_pathology=clinical[:, 2],
                    x_t_stage=clinical[:, 3],
                    x_n_stage=clinical[:, 4],
                    x_m_stage=clinical[:, 5],
                    x_lymph=clinical[:, 6],
                    x_race=clinical[:, 7],
                )
                loss_cls = loss_cls + criterion_cls(hazards=hazards_Omics, S=S_Omics, Y=label, c=censorship)
                loss_div = loss_div + criterion_div(hazards_Omics, hazards_t)
                loss_kd = loss_kd + criterion_kd(feat_Omics, feat_t)
            loss = loss_cls + self.alpha * loss_div + self.beta * loss_kd
            # results
            risk = -torch.sum(S_s, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = censorship.item()
            all_event_times[batch_idx] = data_OS
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # update ema model
            ema_model.update(model)
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print("loss: {:.4f}, c_index: {:.4f}".format(loss, c_index))
        if self.writer:
            self.writer.add_scalar("train/loss", loss, self.epoch)
            self.writer.add_scalar("train/c_index", c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc="Test")
        survival_test = []
        for batch_idx, (data_ID, data_OS, data_Censorship, data_Label, data_WSI, data_Omics, data_Clinical) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                clinical = data_Clinical.type(torch.FloatTensor).cuda()
                label = data_Label.type(torch.LongTensor).cuda()
                censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # for AUC
            survival_test.append((True if data_Censorship[0] == 0 else False, data_OS[0].item()))
            # prediction
            with torch.no_grad():
                hazards, S, features = model(
                    x_WSI=data_WSI,
                    x_Omics=data_Omics,
                    x_age=clinical[:, 0],
                    x_gender=clinical[:, 1],
                    x_pathology=clinical[:, 2],
                    x_t_stage=clinical[:, 3],
                    x_n_stage=clinical[:, 4],
                    x_m_stage=clinical[:, 5],
                    x_lymph=clinical[:, 6],
                    x_race=clinical[:, 7],
                )
            loss = criterion[0](hazards=hazards, S=S, Y=label, c=censorship)
            total_loss += loss.item()
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = censorship.item()
            all_event_times[batch_idx] = data_OS
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print("loss: {:.4f}, c_index: {:.4f}".format(loss, c_index))
        if self.writer and (not self.args.evaluate):
            self.writer.add_scalar("val/loss", loss, self.epoch)
            self.writer.add_scalar("val/c_index", c_index, self.epoch)
        return c_index, survival_test, all_risk_scores

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, "fold_" + str(self.fold), "model_best_{score:.4f}_{epoch}.pth.tar".format(score=state["best_score"], epoch=state["epoch"]))
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)
