import torch
from torch import nn

from .attention import NystromAttention
# from timm.models.vision_transformer import _cfg, Mlp, Block


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class UnimodalBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_modal=3,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33
    ):
        super().__init__()
        self.num_modal = num_modal
        self.layers = nn.ModuleList([])
        for _ in range(num_modal):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            NystromAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                num_landmarks=num_landmarks,
                                pinv_iterations=pinv_iterations,
                                residual=attn_values_residual,
                                residual_conv_kernel=attn_values_residual_conv_kernel
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, num_tokens, mask=None):
        assert len(num_tokens) == self.num_modal, "length of num_tokens list must be equal to num_modal"
        modality1 = x[:, 0:num_tokens[0], :]
        modality2 = x[:, num_tokens[0]:sum(num_tokens[:2]), :]
        modality3 = x[:, sum(num_tokens[:2]):, :]
        tokens = [modality1, modality2, modality3]
        for idx in range(self.num_modal):
            attn, ff = self.layers[idx]
            tokens[idx] = attn(tokens[idx], mask=mask) + tokens[idx]
            tokens[idx] = ff(tokens[idx]) + tokens[idx]
        tokens = torch.cat(tokens, dim=1)
        return tokens


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # print('input:', x.shape)

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        # print('output:', x.shape)
        return x


class CrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_modal=3, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.num_modal = num_modal
        self.layers = nn.ModuleList([])
        for _ in range(num_modal):
            self.layers.append(
                nn.ModuleList(
                    [
                        norm_layer(dim),
                        CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale),
                        norm_layer(dim),
                        Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
                    ]
                )
            )

    def forward(self, x, num_tokens):
        assert len(num_tokens) == self.num_modal, "length of num_tokens list must be equal to num_modal"
        modality1 = x[:, 0:num_tokens[0], :]
        modality2 = x[:, num_tokens[0]:sum(num_tokens[:2]), :]
        modality3 = x[:, sum(num_tokens[:2]):, :]
        # cross attention for modality #1
        norm1, attn, norm2, mlp = self.layers[0]
        cross = modality1[:, 0:1, ...] + attn(norm1(torch.cat((modality1[:, 0:1, ...], modality2, modality3), dim=1)))
        cross = cross + mlp(norm2(cross))
        modality1[:, 0:1, ...] = cross
        # cross attention for modality #2
        norm1, attn, norm2, mlp = self.layers[1]
        cross = modality2[:, 0:1, ...] + attn(norm1(torch.cat((modality2[:, 0:1, ...], modality1, modality3), dim=1)))
        cross = cross + mlp(norm2(cross))
        modality2[:, 0:1, ...] = cross
        # cross attention for modality #3
        norm1, attn, norm2, mlp = self.layers[2]
        cross = modality3[:, 0:1, ...] + attn(norm1(torch.cat((modality3[:, 0:1, ...], modality1, modality2), dim=1)))
        cross = cross + mlp(norm2(cross))
        modality3[:, 0:1, ...] = cross
        tokens = [modality1, modality2, modality3]
        tokens = torch.cat(tokens, dim=1)
        return tokens


class CrossFormer(nn.Module):
    def __init__(self, dim, depth) -> None:
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([UnimodalBlock(dim=dim), CrossModalBlock(dim=dim)]))

    def forward(self, x, num_tokens):
        for uni_modal, cross_modal in self.layers:
            x = uni_modal(x, num_tokens)
            x = cross_modal(x, num_tokens)
        return x
