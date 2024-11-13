import torch
from einops import rearrange
from torch import nn
from typing import Sequence


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = nn.Hardswish(True)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x, resolution):
        H, W = resolution

        x = self.pwconv1(x)
        x = self.act1(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.pwconv2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, _res):
        B, N, C = x.shape
        q = (
            self.wq(x[:, 0:1, ...])
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B1C -> B1H(C/H) -> BH1(C/H)
        k = (
            self.wk(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # BNC -> BNH(C/H) -> BHN(C/H)
        v = (
            self.wv(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # BNC -> BNH(C/H) -> BHN(C/H)
        with torch.no_grad():
            x = (
                nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, dropout_p=self.attn_drop
                )
                .transpose(1, 2)
                .reshape(B, 1, C)
            )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, hidden_features):
        super(DWConv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
                groups=hidden_features,
                bias=False,
            ),
            nn.GELU(),
        )

    def forward(self, x, x_size):
        x = rearrange(x, "b (h w) c -> b c h w", h=x_size[0], w=x_size[1])
        x = self.depthwise_conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class Block(nn.Module):
    def __init__(self, dim: int = 64, att=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.t_mix = (
            SepConv(dim)
            if not att
            else CrossAttention(dim, attn_drop=attn_drop, proj_drop=proj_drop)
        )
        self.mlp = DWConv(dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x, res):
        x = self.t_mix(self.norm1(x), res) + x
        return self.mlp(self.norm2(x), res) + x


class Blocks(nn.Module):
    def __init__(self, dim, n_block, att=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList() + [
            Block(dim, att, attn_drop, proj_drop) for _ in range(n_block)
        ]

    def forward(self, x):
        _B, _C, H, W = x.shape
        x = rearrange(x, "b c h w-> b (h w) c")
        for block in self.blocks:
            x = block(x, (H, W))
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class Down(nn.Sequential):
    def __init__(self, dim, out_dim, stride):
        super().__init__(
            nn.Conv2d(
                dim, out_dim, stride * 2 - 1, stride, (stride * 2 - 1) // 2, bias=False
            ),
            nn.BatchNorm2d(out_dim),
        )


class MetaGan(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        n_class: int = 1,
        dims: Sequence[int] = (48, 96, 192, 288),
        blocks: Sequence[int] = (3, 3, 9, 3),
        attention: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop: float = 0.2,
        sigmoid: bool = True,
    ):
        super().__init__()
        dims = [in_ch] + list(dims)
        states = []
        for index in range(len(blocks)):
            states.append(Down(dims[index], dims[index + 1], 4 if index == 0 else 2))
            states.append(
                Blocks(
                    dims[index + 1],
                    blocks[index],
                    index > 1 if attention else False,
                    attn_drop,
                    proj_drop,
                )
            )
        self.blocks = nn.Sequential(*states)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                dims[-1], dims[-1] * 6, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(dims[-1] * 6),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        to_class = []
        to_class += [
            nn.Linear(dims[-1] * 6, dims[-1] * 8, bias=False),
            nn.BatchNorm1d(dims[-1] * 8),
            nn.Hardswish(inplace=True),
            nn.Dropout(drop),
            nn.Linear(dims[-1] * 8, n_class),
        ]
        if sigmoid:
            to_class += [nn.Hardsigmoid(True)]
        self.to_class = nn.Sequential(*to_class)

    def forward(self, x):
        x = self.blocks(x)
        x = self.conv1(x).flatten(1)
        x = self.to_class(x)
        return x