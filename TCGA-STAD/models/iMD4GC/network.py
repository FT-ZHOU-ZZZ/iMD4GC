import torch

import torch.nn as nn

from .transformer import CrossFormer
from .tokenize import TokenizeRecord, TokenizeWSI, TokenizeOmics
from .fusion import ConcatWithLinear, TensorFusion, MultiplicativeInteractions3Modal, LowRankTensorFusion
from .fusion import EarlyFusionTransformer, LateFusionTransformer


class iMD4GC(nn.Module):
    def __init__(self, pkl, num_classes, n_WSI=1024, fusion="ConcatWithLinear", dim_token=128, layers=2) -> None:
        super(iMD4GC, self).__init__()
        self.fusion = fusion
        self.dim_token = dim_token
        self.num_classes = num_classes
        # tokenization layer
        self.TokenizeRecord = TokenizeRecord(pkl=pkl, dim=dim_token)
        self.TokenizeWSI = TokenizeWSI(pkl=pkl, n_features=n_WSI, dim=dim_token)
        self.TokenizeOmics = TokenizeOmics(pkl=pkl, dim=dim_token)
        # transformer layer
        self.Transformer = CrossFormer(dim=dim_token, depth=layers)
        # fusion Layer
        if self.fusion == "ConcatWithLinear":
            self.ff = ConcatWithLinear(input_dim=3 * dim_token, output_dim=dim_token, concat_dim=1)
        elif self.fusion == "Multiplicative":
            self.ff = MultiplicativeInteractions3Modal(input_dims=(dim_token, dim_token, dim_token), output_dim=dim_token)
        elif self.fusion == "TensorFusion":
            self.ff = TensorFusion(input_dims=(dim_token, dim_token, dim_token), output_dim=dim_token)
        elif self.fusion == "LowRankTensorFusion":
            self.ff = LowRankTensorFusion(input_dims=(dim_token, dim_token, dim_token), output_dim=dim_token, rank=16)
        elif self.fusion == "EarlyFusionTransformer":
            self.ff = EarlyFusionTransformer(n_features=3)
        elif self.fusion == "LateFusionTransformer":
            embed_dim = 9
            self.ff = LateFusionTransformer(embed_dim=embed_dim)
            dim_token = embed_dim
        else:
            raise NotImplementedError("fusion [{}] is not implemented".format(self.fusion))
        # classification layer
        self.classifier = nn.Linear(dim_token, self.num_classes)

    def forward(self, **kwargs):
        # construct tokens for each modality
        tokens_clin = self.TokenizeRecord(**kwargs)
        tokens_path = self.TokenizeWSI(**kwargs)
        tokens_omic = self.TokenizeOmics(**kwargs)
        tokens = torch.cat((tokens_clin, tokens_path, tokens_omic), dim=1)
        num_tokens = [tokens_clin.shape[1], tokens_path.shape[1], tokens_omic.shape[1]]
        #
        tokens = self.Transformer(tokens, num_tokens)
        # fetch cls token for each modality
        cls_clin = tokens[:, 0, :]
        cls_path = tokens[:, tokens_clin.shape[1], :]
        cls_omic = tokens[:, tokens_clin.shape[1] + tokens_path.shape[1], :]

        # feature fusion
        feature = self.ff([cls_clin, cls_path, cls_omic])
        # prediction
        logits = self.classifier(feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, feature
