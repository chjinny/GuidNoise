import torch
from functools import partial
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from denoising_diffusion_pytorch import denoising_diffusion_pytorch as base

class base_unet(base.Unet):
    def __init__(self, dim, channels, dim_mults, flash_attn=False, init_dim = None, dropout = 0., attn_heads = 4, full_attn = None, attn_dim_head = 32, learned_sinusoidal_dim = 16, sinusoidal_pos_emb_theta = 10000, random_fourier_features = False, **kwargs):
        super().__init__(dim=dim, channels=channels, dim_mults=dim_mults, flash_attn=flash_attn, dropout=dropout, init_dim = init_dim, attn_heads = attn_heads, attn_dim_head = attn_dim_head, learned_sinusoidal_dim = learned_sinusoidal_dim, sinusoidal_pos_emb_theta = sinusoidal_pos_emb_theta, random_fourier_features = random_fourier_features, **kwargs)
        self.dim = dim
        self.time_dim = dim * 2
        self.cond_dim = dim * 4
        init_dim = base.default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = base.cast_tuple(full_attn, num_stages)
        attn_heads = base.cast_tuple(attn_heads, num_stages)
        attn_dim_head = base.cast_tuple(attn_dim_head, num_stages)

        FullAttention = partial(base.Attention, flash = flash_attn)
        resnet_block = partial(base.ResnetBlock, time_emb_dim = self.cond_dim, dropout = dropout)
        self.ups = torch.nn.ModuleList([])
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else base.LinearAttention

            self.ups.append(torch.nn.ModuleList([
                resnet_block(dim_out + dim_in*2, dim_out),
                resnet_block(dim_out + dim_in*2, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                base.Upsample(dim_out, dim_in) if not is_last else  torch.nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.final_res_block = resnet_block(init_dim * 3, init_dim)

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = base.RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = base.SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim
        self.time_mlp = torch.nn.Sequential(
            sinu_pos_emb,
            torch.nn.Linear(fourier_dim, self.time_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.time_dim, self.time_dim)
        )

        self.noise_aware_module = torch.nn.ModuleList([
            resnet_block(dims[-1]*2, dims[-1]),
            resnet_block(dims[-1], dims[-1]),
            attn_klass(dims[-1]),
            resnet_block(dims[-1], dims[-1]),
            resnet_block(dims[-1], dims[-1]),
            attn_klass(dims[-1]),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(1),
            torch.nn.Linear(dims[-1], self.time_dim),
        ])        

    def forward(self, diffusion_noise, clean_image, refer_noisy, refer_clean, time):
        assert all([base.divisible_by(d, self.downsample_factor) for d in diffusion_noise.shape[-2:]]), f'your input dimensions {diffusion_noise.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        b, _, h, w = diffusion_noise.shape
        x = torch.cat((
            diffusion_noise.unsqueeze(1),
            clean_image.unsqueeze(1),
            refer_noisy.unsqueeze(1),
            refer_clean.unsqueeze(1),
        ), dim = 1).reshape(b*4, -1, h, w)
        
        x = self.init_conv(x)
        rx = x.clone().reshape(b, 4, *x.shape[1:])[:,:2].reshape(b*2, *x.shape[1:])

        t = self.time_mlp(time)

        t = torch.cat([t.unsqueeze(1)]*4, dim = 1).reshape(b*4, self.time_dim)
        t = torch.cat([t, t], dim = -1)

        hx = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            hx.append(x.reshape(b, 4, *x.shape[1:])[:,:2].reshape(b*2, *x.shape[1:]))
            x = block2(x, t)
            x = attn(x) + x
            hx.append(x.reshape(b, 4, *x.shape[1:])[:,:2].reshape(b*2, *x.shape[1:]))
            x = downsample(x)

        fb, fc, fh, fw = x.shape
        x = x.reshape(b, 4, fc, fh, fw)
        reference_embd = torch.cat([x[:, 2], x[:,3]],dim=1)
        x = x[:, 0]
        t = t.reshape(b, 4, -1)
        t = t[:, 0]
        for layer in self.noise_aware_module:
            if isinstance(layer, base.ResnetBlock):
                reference_embd = layer(reference_embd, t)
            elif isinstance(layer, base.Attention):
                reference_embd = layer(reference_embd) + reference_embd
            else:
                reference_embd = layer(reference_embd)
        t = t[:, :self.time_dim]
        t = torch.cat([t, reference_embd], dim=-1)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            px = hx.pop()
            px = px.reshape(b, 2*px.shape[1], *px.shape[-2:])
            x = torch.cat((x, px), dim = 1)
            x = block1(x, t)

            px = hx.pop()
            px = px.reshape(b, 2*px.shape[1], *px.shape[-2:])
            x = torch.cat((x, px), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, rx.reshape(b, 2*rx.shape[1], *rx.shape[-2:])), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    