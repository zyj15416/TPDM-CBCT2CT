import copy
import math
from functools import partial
from einops import rearrange
from torch import einsum
from torch.fft import fft2, ifft2
from timm.models.layers import trunc_normal_, DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F


class build(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.map4 = nn.Sequential(
            nn.Conv2d(32, out_channel, 1, 1)
        )

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        return output4


class LayerNorm_1(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class ResnetBlock(nn.Module):
    class Block(nn.Module):
        def __init__(self, dim, dim_out, groups=8):
            super().__init__()
            self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
            self.norm = nn.GroupNorm(groups, dim_out)
            self.act = nn.SiLU()

        def forward(self, x, scale_shift=None):
            # print(x.de)
            x = self.proj(x)
            x = self.norm(x)

            if exists(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift

            x = self.act(x)
            return x

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, num_res_blocks=2):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.init_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.block1_all = []
        self.block2_all = []
        self.res_conv_all = []
        for i in range(num_res_blocks):
            self.block1_all.append(self.Block(dim, dim_out, groups=groups) if
                                   i == 0 else
                                   self.Block(dim_out, dim_out, groups=groups))
            self.block2_all.append(self.Block(dim_out, dim_out, groups=groups))
            self.res_conv_all.append(nn.Conv2d(dim, dim_out, 1) if dim != dim_out and i == 0 else nn.Identity())
        self.block1_all = nn.ModuleList([*self.block1_all])
        self.block2_all = nn.ModuleList([*self.block2_all])
        self.res_conv_all = nn.ModuleList([*self.res_conv_all])

    def _forward(self, x, time_emb=None, i=0):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1_all[i](x, scale_shift=scale_shift)

        h = self.block2_all[i](h)

        return h + self.res_conv_all[i](x)

    def forward(self, x, time_emb=None):
        # x_copy = x
        # x = self.init_conv(x)
        for i in range(self.num_res_blocks):
            x = self._forward(x, time_emb, i)
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h.to(x.device)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class Conditioning_1(nn.Module):
    class LayerNorm(nn.Module):
        def __init__(self, dim, bias=False):
            super().__init__()
            self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

        def forward(self, x):
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3
            var = torch.var(x, dim=1, unbiased=False, keepdim=True)
            mean = torch.mean(x, dim=1, keepdim=True)
            return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

    def __init__(self, fmap_size, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim * 4, fmap_size // 2, fmap_size // 2))

        self.norm_input = self.LayerNorm(dim, bias=True)
        self.norm_condition = self.LayerNorm(dim, bias=True)

        self.block = ResnetBlock(dim, dim)
        self.dwt = DWT()
        self.iwt = IWT()

    def forward(self, x, c):
        x = self.dwt(x)
        x = x * self.ff_parser_attn_map
        x = self.iwt(x)

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        return self.block(c)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm_1(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):

    def __init__(self, dim, time_emb_dim=None, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        scale_shift = None
        # print(x.shape)
        # print(x[0])
        x, time_emb_init = x

        # print(x.shape)
        # print(time_emb.shape)
        if exists(self.mlp) and exists(time_emb_init):
            time_emb = self.mlp(time_emb_init)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        input = x
        x = self.dwconv(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return [x, time_emb_init]


class fusion(nn.Module):
    def __init__(self, channel):
        super(fusion, self).__init__()
        self.norm = LayerNorm_1(channel, bias=True)

        self.block = ResnetBlock(channel, channel)
        self.scale = torch.sqrt(channel)

    def forward(self, x, y):
        q, k = x
        v = y
        weights = q @ k.transpose(1, 2)
        weights = weights.softmax(1) / self.scale
        out = v @ weights

        out = self.norm(out)
        return self.block(out)


class Build(nn.Module):

    def __init__(self, image_size=256, in_chans=1, depths=[3, 3, 6, 3], dims=[96, 192, 384, 768], self_condition=False,
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], skip_connect_c=False,
                 ):
        super().__init__()
        self.self_condition = self_condition
        self.init_conv = nn.Conv2d(1, dims[0], 7, padding=3)
        self.cond_init_conv = nn.Conv2d(in_chans, dims[0], 7, padding=3)
        self.skip_connect_c = skip_connect_c
        self.run = build(1, 1)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dims[0]),
            nn.Linear(dims[0], dims[0] * 4),
            nn.GELU(),
            nn.Linear(dims[0] * 4, dims[0] * 4)
        )
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.downsample_layers_cond = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        stem_cond = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        self.downsample_layers_cond.append(stem_cond)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers_cond.append(downsample_layer)
        self.upsample_layers = nn.ModuleList()
        for i in range(3, 0, -1):
            upsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.ConvTranspose2d(dims[i], dims[i - 1], kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)

        stem = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=2, stride=2),
        )
        self.upsample_layers.append(stem)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages_cond = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        cur_cond = 0
        self.conditioners = nn.ModuleList([])
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], time_emb_dim=dims[0] * 4, drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.atten = Residual(Attention(dims[-1]))
        self.stages.append(Block(dim=dims[-1], time_emb_dim=dims[0] * 4, drop_path=.1,
                                 layer_scale_init_value=layer_scale_init_value))
        in_size = image_size
        for i in range(4):
            in_size = in_size // 2
            stage = nn.Sequential(
                *[Block(dim=dims[i], time_emb_dim=dims[0] * 4, drop_path=dp_rates[cur_cond + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.conditioners.append(Conditioning_1(in_size, dims[i]))
            self.stages_cond.append(stage)
            cur_cond += depths[i]
        self.stages_cond.append(Block(dim=dims[-1], time_emb_dim=dims[0] * 4, drop_path=.1,
                                      layer_scale_init_value=layer_scale_init_value))
        self.out_indices = out_indices

        self.d_stage = nn.ModuleList([])
        cur_cond = 0
        for i in range(3, -1, -1):
            stage = nn.Sequential(
                *[Block(dim=dims[i], time_emb_dim=dims[0] * 4, drop_path=dp_rates[cur_cond + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.d_stage.append(stage)
            cur_cond += depths[i]
        self.final_conv = nn.Conv2d(dims[0] * 2, 1, 1)

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        norm_layer_cond = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer_cond(dims[i_layer])
            layer_name = f'norm_cond{i_layer}'
            self.add_module(layer_name, layer)
        self.final_res_block = Block(dim=dims[0] * 2, time_emb_dim=dims[0] * 4, drop_path=.3,
                                     layer_scale_init_value=layer_scale_init_value)
        self.final_res_block_cond = Block(dim=dims[0], time_emb_dim=dims[0] * 2, drop_path=.3,
                                          layer_scale_init_value=layer_scale_init_value)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward_features(self, x, t, c):
        outs = []
        h = []
        x = self.init_conv(x)
        r = x.clone()
        c_ = self.run(c)

        c = self.cond_init_conv(c)
        t = self.time_mlp(t)
        for i in range(4):
            # print(x.shape)
            x = self.downsample_layers[i](x)
            x = self.stages[i]([x, t])
            x = x[0]

            c = self.downsample_layers_cond[i](c)
            c = self.stages_cond[i]([c, t])
            c = c[0]
            c = self.conditioners[i](x, c)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
            if i != 3:
                h.append([x, c] if self.skip_connect_c else [x])

        x = self.stages[-1]([x, t])[0]
        c = self.stages_cond[-1]([c, t])[0]

        x = x + c
        x = self.atten(x)
        for i in range(3):
            x = self.d_stage[i]([x, t])[0]
            x = self.upsample_layers[i](x)
            x = torch.add(x, *h.pop())
        x = self.d_stage[-1]([x, t])[0]
        x = self.upsample_layers[-1](x)

        x = torch.concat([x, r], dim=1)
        x = self.final_res_block([x, t])[0]
        return self.final_conv(x), c_

    def forward(self, x, t, x_cond):
        x = self.forward_features(x, t, x_cond)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def model_T(**kwargs) -> Build:
    return Build(dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], **kwargs)


def model_S(**kwargs) -> Build:
    return Build(dims=[96, 192, 384, 768], depths=[3, 3, 27, 3], **kwargs)


def model_B(**kwargs) -> Build:
    return Build(dims=[128, 256, 512, 1024], depths=[3, 3, 27, 3], **kwargs)


if __name__ == "__main__":
    model = model_S().to('cpu')
    x = model(torch.randn(size=(1, 1, 256, 256), device='cpu'),
              torch.randn(size=(1,), device='cpu'),
              torch.randn(size=(1, 1, 256, 256), device='cpu'))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % total)
