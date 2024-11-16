import torch
from einops import rearrange
from torch import nn
from typing import Sequence, Literal

SE_MODS = Literal["SSE", "CSE", "CSSE"]


class CSELayer(nn.Module):
    def __init__(self, num_channels: int = 48, reduction_ratio: int = 2):
        super(CSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.squeezing = nn.Sequential(
            nn.Linear(num_channels, num_channels_reduced, bias=True),
            nn.ReLU(True),
            nn.Linear(num_channels_reduced, num_channels, bias=True),
            nn.Hardsigmoid(True),
        )

    def forward(self, input_tensor):
        squeeze_tensor = torch.mean(input_tensor, dim=1, keepdim=True)
        output_tensor = input_tensor * self.squeezing(squeeze_tensor)
        return output_tensor


class SSELayer(nn.Module):
    def __init__(self, dim: int = 48):
        super().__init__()
        self.squeezing = nn.Sequential(nn.Linear(dim, 1), nn.Hardsigmoid(True))

    def forward(self, x):
        return x * self.squeezing(x)


class CSSELayer(nn.Module):
    def __init__(self, dim: int = 48):
        super().__init__()
        self.sse = SSELayer(dim)
        self.cse = CSELayer(dim)

    def forward(self, x):
        return torch.max(self.sse(x), self.cse(x))


class SepConv(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 2,
        bias: bool = False,
        kernel_size: int = 7,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = nn.Hardswish(True)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
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


class DWConv(nn.Module):
    def __init__(self, hidden_features: int = 48):
        super().__init__()
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
            nn.Mish(True),  # https://arxiv.org/pdf/1908.08681
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = (
            x.transpose(1, 2)
            .view(x.shape[0], self.hidden_features, x_size[0], x_size[1])
            .contiguous()
        )  # b Ph*Pw c
        x = self.depthwise_conv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class FFN(nn.Module):
    def __init__(
        self,
        in_features: int = 48,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.act = nn.Mish(True)  # https://arxiv.org/pdf/1908.08681
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.dwconv = DWConv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(
        self,
        dim: int = 48,
        head_dim: int = 32,
        bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.num_heads = dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=bias)
        self.attn_drop = attn_drop
        self.proj = nn.Sequential(
            nn.Linear(self.attention_dim, dim, bias=bias), nn.Dropout(proj_drop)
        )

    def forward(self, x, _):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        with torch.no_grad():
            nn.functional.scaled_dot_product_attention(
                q, k, v, scale=self.scale, dropout_p=self.attn_drop
            ).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        att: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        head_dim: int = 32,
        mlp_ratio: float = 2.0,
        se_mod: SE_MODS = "SSE",
    ):
        super().__init__()
        self.t_mix = (
            SepConv(dim)
            if not att
            else Attention(dim, head_dim, False, attn_drop, proj_drop)
        )
        self.c_mix = FFN(dim, mlp_ratio)
        if se_mod == "SSE":
            self.se = SSELayer(dim)
        elif se_mod == "CSE":
            self.se = CSELayer(dim)
        else:
            self.se = CSSELayer(dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.gamma1 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x, res):
        x = self.gamma1 * self.t_mix(self.norm1(x), res) + x
        x = self.gamma2 * self.c_mix(self.norm2(x), res) + x
        return self.se(x)


class Blocks(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        n_block: int = 2,
        att: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        head_dim: int = 32,
        mlp_ratio: float = 2.0,
        se_mod: SE_MODS = "sse",
    ):
        super().__init__()
        self.blocks = nn.ModuleList() + [
            Block(dim, att, attn_drop, proj_drop, head_dim, mlp_ratio, se_mod)
            for _ in range(n_block)
        ]
        self.short = nn.Linear(dim * 2, dim)

    def forward(self, x):
        _B, _C, H, W = x.shape
        x = rearrange(x, "b c h w-> b (h w) c")
        short = x
        for block in self.blocks:
            x = block(x, (H, W))
        x = self.short(torch.cat([x, short], -1))
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class DownPU(nn.Sequential):
    def __init__(self, dim: int = 48, out_dim: int = 48, down: int = 2):
        stages = [
            nn.Conv2d(dim, out_dim // (down * down), 3, 1, 1, bias=False),
            nn.PixelUnshuffle(down),
            nn.GroupNorm(4, out_dim),
        ]
        super().__init__(*stages)


class MlpHead(nn.Module):
    def __init__(
        self, dim: int = 48, num_classes: int = 1000, head_dropout: float = 0.0
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim * 6, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, dim * 6),
            nn.Hardswish(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_head = nn.Sequential(
            nn.Linear(dim * 6, dim * 8, bias=False),
            nn.RMSNorm(dim * 8),
            nn.Hardswish(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(dim * 8, num_classes),
        )

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.to_head(x)


class MetaGan(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        n_class: int = 1,
        dims: Sequence[int] = (48, 96, 192, 288),
        blocks: Sequence[int] = (3, 3, 9, 3),
        downs: Sequence[int] = (4, 4, 2, 2),
        se_mode: SE_MODS = "SSE",  # cse, csse, sse
        mlp_ratio: float = 2.0,
        attention: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        head_dim: int = 32,
        drop: float = 0.2,
        sigmoid: bool = False,
    ):
        super().__init__()
        dims = [in_ch] + list(dims)
        states = []
        for index in range(len(blocks)):
            states.append(DownPU(dims[index], dims[index + 1], downs[index]))
            states.append(
                Blocks(
                    dims[index + 1],
                    blocks[index],
                    index > 1 if attention else False,
                    attn_drop,
                    proj_drop,
                    head_dim,
                    mlp_ratio,
                    se_mode,
                )
            )

        states.append(MlpHead(dims[-1], n_class, head_dropout=drop))
        if sigmoid:
            states.append(nn.Hardsigmoid(True))
        self.stages = nn.Sequential(*states)

    def forward(self, x):
        return self.stages(x)
