import pickle
import math
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

records = ["age", "gender", "pathology", "t_stage", "n_stage", "m_stage", "lymph", "race"]


class TokenizeRecord(nn.Module):
    def __init__(self, pkl, dim=256) -> None:
        super(TokenizeRecord, self).__init__()
        # load word embedding
        wv = pickle.load(open(pkl, "rb"))
        for key, val in wv.items():
            wv[key] = torch.from_numpy(val).to("cuda" if torch.cuda.is_available() else "cpu").float()
        self.wv = wv
        # embedding layer for each clinical indicator
        fc_Age = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Gender = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Pathology = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Tstage = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Nstage = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Mstage = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Lymph = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        fc_Race = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        clinical_net = [fc_Age, fc_Gender, fc_Pathology, fc_Tstage, fc_Nstage, fc_Mstage, fc_Lymph, fc_Race]
        self.record_net = nn.ModuleList(clinical_net)
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, **kwargs):
        bs = 1
        cls_token = self.cls_token.repeat(bs, 1, 1).cuda()
        cls_token = self.wv["clinical"].repeat(bs, 1, 1) + cls_token
        tokens = cls_token
        for idx in range(len(records)):
            key = records[idx]
            val = kwargs["x_" + key]
            if torch.sum(val) >= 0:
                wv = self.wv[key.replace("_", "-")]
                token = wv.view(1, 1, -1) + self.record_net[idx](val)
                token = self.wv["clinical"].repeat(bs, 1, 1) + token
                tokens = torch.cat((tokens, token), dim=1)
        # print('clinical tokens', tokens.shape)
        return tokens


class TokenizeWSI(nn.Module):
    def __init__(self, pkl, dim=256, n_features=1024) -> None:
        super(TokenizeWSI, self).__init__()
        # load word embedding
        wv = pickle.load(open(pkl, "rb"))
        for key, val in wv.items():
            wv[key] = torch.from_numpy(val).to("cuda" if torch.cuda.is_available() else "cpu").float()
        self.wv = wv
        # embedding layer
        self.WSI_net = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(), nn.Linear(256, dim), nn.ReLU())
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, **kwargs):
        bs = 1
        x_WSI = kwargs["x_WSI"]
        cls_token = self.cls_token.repeat(bs, 1, 1).cuda()
        cls_token = self.wv["pathology"].repeat(bs, 1, 1) + cls_token
        tokens = cls_token
        if x_WSI.shape[1] > 1:
            x_WSI = self.WSI_net(x_WSI)
            x_WSI = self.wv["pathology"].repeat(bs, x_WSI.shape[1], 1) + x_WSI
            tokens = torch.cat((tokens, x_WSI), dim=1)
        # print('WSI tokens', tokens.shape)
        return tokens


class TokenizeOmics(nn.Module):
    def __init__(self, pkl, dim=256) -> None:
        super(TokenizeOmics, self).__init__()
        # load word embedding
        wv = pickle.load(open(pkl, "rb"))
        for key, val in wv.items():
            wv[key] = torch.from_numpy(val).to("cuda" if torch.cuda.is_available() else "cpu").float()
        self.wv = wv
        # load word embedding for all genes
        gene2vec = np.load("./word2vec/gene2vec.npy")
        gene2vec = torch.from_numpy(gene2vec).float()
        self.emb = nn.Embedding.from_pretrained(gene2vec)
        # gene embedding layer
        self.fc_RNA = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, **kwargs):
        Omics = kwargs["x_Omics"]
        bs = 1
        cls_token = self.cls_token.repeat(bs, 1, 1).cuda()
        cls_token = self.wv["omics"].repeat(bs, 1, 1) + cls_token
        tokens = cls_token
        if len(Omics) > 1:
            index = Omics[1].to("cuda" if torch.cuda.is_available() else "cpu").int()
            index = self.emb(index)
            value = Omics[2].to("cuda" if torch.cuda.is_available() else "cpu").float()
            value = self.fc_RNA(value.view(bs, -1, 1))
            token = index + value
            token = self.wv["omics"].repeat(bs, token.shape[1], 1) + token
            tokens = torch.cat((tokens, token), dim=1)
        # print('Omics tokens', tokens.shape)
        return tokens
