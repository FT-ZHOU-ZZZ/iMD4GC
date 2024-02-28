import random

import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data


class TCGA_Survival(data.Dataset):
    def __init__(self, excel_file, modal, keep_missing=True):
        self.modal = modal
        self.omics_size = [14170]
        self.keep_missing = keep_missing
        print("[dataset] loading dataset from %s" % (excel_file))
        rows = pd.read_excel(excel_file)
        rows = self.disc_label(rows).values.tolist()
        print("[dataset] required modality : %s" % (modal))
        for key in modal.split("_"):
            if key not in ["Clinical", "WSI", "Omics"]:
                raise NotImplementedError("modality [{}] is not implemented".format(modal))
        self.cases = []
        for row in rows:
            digital = self.__digitize__(row)
            if "Clinical" in self.modal:
                self.cases.append(digital)
            elif "WSI" in self.modal:
                if digital[4]:
                    self.cases.append(digital)
            elif "Omics" in self.modal:
                if digital[5]:
                    self.cases.append(digital)
            else:
                raise NotImplementedError("modality [{}] is not implemented".format(modal))
        if not self.keep_missing:
            if len(modal.split("_")) > 1:
                missing = []
                for idx in range(len(self.cases)):
                    ID, OS, Censorship, Label, WSI, Omics, Clinical = self.cases[idx]
                    if ("WSI" in modal) and (WSI is None):
                        missing.append(idx)
                    elif ("Omics" in modal) and (Omics is None):
                        missing.append(idx)
                self.cases = [self.cases[idx] for idx in range(len(self.cases)) if idx not in missing]
            print("[dataset] dataset from %s, number of cases=%d (exclude modality-incomplete data)" % (excel_file, len(self.cases)))
        else:
            print("[dataset] dataset from %s, number of cases=%d (include modality-incomplete data)" % (excel_file, len(self.cases)))

    def __func__(self, arg):
        ID, OS, Censorship, Label, WSI, Omics, Clinical = arg
        if ("WSI" in self.modal) and (WSI is not None):
            WSI = self.__WSI__(WSI)
        else:
            WSI = torch.zeros((1))
        if ("Omics" in self.modal) and (Omics is not None):
            Omics = self.__Omics__(Omics)
        else:
            Omics = torch.zeros((1))
        if "Clinical" in self.modal:
            Clinical = torch.from_numpy(np.array(Clinical)).to(torch.float32)
        else:
            Clinical = torch.zeros((1))
        return [ID, OS, Censorship, Label, WSI, Omics, Clinical]

    def __Omics__(self, file):
        # read RNA sequence
        data = pd.read_csv(file)
        key = data.iloc[:, 0].values.tolist()
        index = torch.from_numpy(np.array(data.iloc[:, 2].values.tolist()).astype(np.int32))
        value = torch.from_numpy(np.array(data.iloc[:, 3].values.tolist()).astype(np.float32))
        return (key, index, value)

    def __WSI__(self, path):
        path = path.replace("resnet50-512", "ctranspath")
        wsi = [torch.load(x) for x in path.split(";")]
        wsi = torch.cat(wsi, dim=0)
        return wsi

    def __digitize__(self, row):
        ID = str(row[0])
        OS = float(row[1])
        Censorship = 1 if int(row[2]) == 0 else 0
        Label = row[-1]
        WSI = str(row[3]) if str(row[3]) != "nan" else None
        # Omics
        # CNV = str(row[4]) if str(row[4]) != 'nan' else None
        RNAseq = str(row[6]) if str(row[6]) != "nan" else None
        # Mutation = str(row[8]) if str(row[8]) != 'nan' else None
        # Clinical
        age = float(row[10]) if str(row[10]) != "nan" else -1
        gender = float(["male", "female"].index(row[11])) if str(row[11]) != "nan" else -1
        pathology = float(["stagei", "stageii", "stageiii", "stageiv"].index(row[12])) if str(row[12]) != "nan" else -1
        T_stage = float(["t1", "t2", "t3", "t4"].index(row[13])) if str(row[13]) != "nan" else -1
        N_stage = float(["n0", "n1", "n2", "n3"].index(row[14])) if str(row[14]) != "nan" else -1
        M_stage = float(["m0", "m1"].index(row[15])) if str(row[15]) != "nan" else -1
        lymph = float(row[16]) if str(row[16]) != "nan" else -1
        race = float(["white", "asian", "blackorafricanamerican"].index(row[17])) if str(row[17]) != "nan" else -1
        Omics = RNAseq
        Clinical = [age, gender, pathology, T_stage, N_stage, M_stage, lymph, race]
        return [ID, OS, Censorship, Label, WSI, Omics, Clinical]

    def get_split(self, fold=0, ratio=0.2, seed=1):
        random.seed(1)
        assert 0 <= fold <= 1 / ratio - 1, "fold should be in 0 ~ {}".format(1 / ratio - 1)
        sample_index = random.sample(range(len(self.cases)), len(self.cases))
        num_split = round((len(self.cases) - 1) * ratio)
        if fold < 1 / ratio - 1:
            val_split = sample_index[fold * num_split : (fold + 1) * num_split]
        else:
            val_split = sample_index[fold * num_split :]
        train_split = [i for i in sample_index if i not in val_split]
        print("[dataset] training split: {}, validation split: {}".format(len(train_split), len(val_split)))
        return train_split, val_split

    def __getitem__(self, index):
        ID, OS, Censorship, Label, WSI, Omics, Clinical = self.cases[index]
        ID, OS, Censorship, Label, WSI, Omics, Clinical = self.__func__((ID, OS, Censorship, Label, WSI, Omics, Clinical))
        return (ID, OS, Censorship, Label, WSI, Omics, Clinical)

    def __len__(self):
        return len(self.cases)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows["Dead"] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df["OS"], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows["OS"].max() + eps
        q_bins[0] = rows["OS"].min() - eps
        disc_labels, q_bins = pd.cut(rows["OS"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        print(q_bins)
        rows.insert(len(rows.columns), "Label", disc_labels.values.astype(int))
        return rows


if __name__ == "__main__":
    from torch.utils.data import DataLoader, SubsetRandomSampler

    random.seed(1)
    dataset = TCGA_Survival("/data/GastricCancer/TCGA-STAD/All_Statistics.xls", modal="Clinical_WSI", keep_missing=False)
    # train_split, test_split = dataset.get_split(fold=4)
    # train_loader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(train_split), num_workers=8)
    # for batch_idx, (data_ID, data_OS, data_Censorship, data_Label, data_WSI, data_Omics, data_Clinical) in enumerate(tqdm(train_loader)):
    #     CNV, RNA, MUT = data_Omics
    #     print(len(CNV))
    # print(len(data_Omics))

    # train_split, test_split = dataset.get_split(fold=4)
    # train_loader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(train_split), num_workers=8)
    # test_loader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(test_split), num_workers=8)
    # for batch_idx, (data_Center, data_ID, data_OS, data_Censorship, data_Label, data_WSI, data_CT, data_Clinical) in (enumerate(tqdm(train_loader))):
    #     pass
