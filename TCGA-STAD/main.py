import os
import time
import numpy as np

from dataset.TCGA_Survival import TCGA_Survival

from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import CV_Meter

from torch.utils.data import DataLoader, SubsetRandomSampler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    if args.evaluate:
        results_dir = args.resume
    else:
        # create results directory
        model_dir = args.model
        if args.layers is not None:
            model_dir += "-{}".format(args.layers)
        if args.alpha is not None:
            model_dir += "-{}".format(args.alpha)
        if args.beta is not None:
            model_dir += "-{}".format(args.beta)
        ckpt_dir = "[{}]".format(args.loss)
        if args.fusion:
            ckpt_dir += "-[{}]".format(args.fusion)
        ckpt_dir += "-[{}]".format(time.strftime("%Y-%m-%d]-[%H-%M-%S"))
        results_dir = "./results/{modal}/{model}/{ckpt}".format(modal=args.modal, model=model_dir, ckpt=ckpt_dir)
    print(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    dataset = TCGA_Survival(excel_file=args.excel_file, modal=args.modal)
    args.num_classes = 4
    # 5-fold cross validation
    meter = CV_Meter(fold=5)
    AUC = []
    # start 5-fold CV evaluation.
    for fold in range(5):
        # get split
        train_split, val_split = dataset.get_split(fold, ratio=0.2, seed=args.seed)
        set_seed(args.seed)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=SubsetRandomSampler(train_split))
        val_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=SubsetRandomSampler(val_split))

        # build model, criterion, optimizer, schedular
        if args.model == "MissGastircKD":
            from models.iMD4GC.network import iMD4GC
            from models.iMD4GC.engine import Engine

            model = iMD4GC(pkl="word2vec/bio_word2vec.pkl", num_classes=args.num_classes, n_WSI=args.n_features, dim_token=200, fusion=args.fusion, layers=args.layers)
            engine = Engine(args, results_dir, fold, alpha=args.alpha, beta=args.beta)
        # Complete Modality
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)
        # start training
        if args.evaluate:
            score, mean_auc = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
            AUC.append(mean_auc)
        else:
            score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
            meter.updata(score, epoch)
    if not args.evaluate:
        csv_path = os.path.join(results_dir, "results_{}.csv".format(args.model))
        meter.save(csv_path)
    else:
        print("AUC: ", AUC)
        print("mean AUC: ", np.mean(AUC))
        print("std AUC: ", np.std(AUC))


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
