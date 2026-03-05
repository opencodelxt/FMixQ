"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]

class SA_WFFCM(nn.Module):

    def __init__(self,
                 in_ch,
                 mid_ch=None,
                 win_size=8,
                 groups=6,
                 enable_freq=True,
                 enable_spatial=True,
                 enable_fuse_conv=True,
                 enable_cfse=True):
        super().__init__()
        mid_ch = mid_ch or in_ch
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.win = int(win_size)
        self.groups = int(groups)
        self.enable_freq = bool(enable_freq)
        self.enable_spatial = bool(enable_spatial)
        self.enable_fuse_conv = bool(enable_fuse_conv)
        self.enable_cfse = bool(enable_cfse)


        self.reduce = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.act = nn.GELU()


        self.spatial = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 1),
            nn.BatchNorm2d(mid_ch),
        )


        if enable_freq:

            self.groups = max(1, min(groups, mid_ch))
            self.freq_pw = nn.ModuleList([

                nn.Conv2d((mid_ch // self.groups) * 2, (mid_ch // self.groups) * 2, 1)
                for _ in range(self.groups)
            ])
            self.freq_bn = nn.BatchNorm2d(mid_ch)
        else:
            self.freq_pw = None


        self.cross_pw = nn.Conv2d(mid_ch, mid_ch, 1)
        self.cross_bn = nn.BatchNorm2d(mid_ch)


        self.cf_fc1 = nn.Linear(mid_ch, max(4, mid_ch // 4))
        self.cf_fc2 = nn.Linear(max(4, mid_ch // 4), mid_ch)


        self.output = nn.Conv2d(mid_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        mid = self.reduce(x)


        if self.enable_spatial:
            s = self.spatial(mid)
        else:

            s = torch.zeros_like(mid)

        if not self.enable_freq:

            f_recon = torch.zeros_like(mid)
        else:
            win = self.win
            stride = win

            pad_h = (win - H % win) % win
            pad_w = (win - W % win) % win
            mid_p = F.pad(mid, (0, pad_w, 0, pad_h))


            unfold = F.unfold(mid_p, kernel_size=win, stride=stride)
            B_, Cw, L = unfold.shape
            patches = unfold.view(B, self.mid_ch, win, win, L).permute(0, 4, 1, 2, 3)
            patches = patches.reshape(B * L, self.mid_ch, win, win)

            group_ch = self.mid_ch // self.groups
            out_groups = []

            for g in range(self.groups):
                start, end = g * group_ch, (g + 1) * group_ch
                wg = patches[:, start:end, :, :]  # (B*L, gch, win, win)
                Fcomp = torch.fft.rfft2(wg, norm='ortho')
                Fr, Fi = Fcomp.real, Fcomp.imag
                Fcat = torch.cat([Fr, Fi], dim=1)
                Fproc = self.freq_pw[g](Fcat)

                half = Fproc.shape[1] // 2

                Fr2 = Fproc[:, :half, :, :]
                Fi2 = Fproc[:, half:2*half, :, :]
                Frec = torch.complex(Fr2, Fi2)
                # iFFT2：恢复到时域
                rec = torch.fft.irfft2(Frec, s=(win, win), norm='ortho')
                out_groups.append(rec)

            freq_patches = torch.cat(out_groups, dim=1)  # (B*L, mid_ch, win, win)
            freq_patches = freq_patches.view(B, L, self.mid_ch, win, win)
            freq_patches = freq_patches.permute(0, 2, 3, 4, 1).reshape(B, self.mid_ch * win * win, L)
            f_recon = F.fold(freq_patches, output_size=(H + pad_h, W + pad_w),
                             kernel_size=win, stride=stride)
            f_recon = f_recon[:, :, :H, :W]
            f_recon = self.freq_bn(f_recon)

        # --- Fusion & CF-SE ---
        fused = s + f_recon
        if self.enable_fuse_conv:
            fused = self.cross_bn(self.cross_pw(fused))
            fused = self.act(fused)
        if self.enable_cfse:
            gap = fused.mean(dim=(2, 3))
            se = self.act(self.cf_fc1(gap))
            se = torch.sigmoid(self.cf_fc2(se)).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            fused = fused * se
        out = fused

        out = self.output(out)
        return x + out


class CrossScaleFreqAttention(nn.Module):

    def __init__(self, in_ch, heads=4, proj_dim=32):
        super().__init__()
        self.heads = heads
        self.proj = nn.Linear(in_ch, proj_dim)
        self.q_proj = nn.Linear(proj_dim, proj_dim)
        self.k_proj = nn.Linear(proj_dim, proj_dim)
        self.v_proj = nn.Linear(proj_dim, proj_dim)
        self.out = nn.Linear(proj_dim, in_ch)

    def forward(self, target_win, neighbor_wins):

        B, L, C, w, _ = target_win.shape
        # 1) 提取窗口级 token
        tgt_token = target_win.mean(dim=(3, 4))  # (B, L, C)
        neigh_tokens = [n.mean(dim=(3, 4)) for n in neighbor_wins]  # [(B, L, C), ...]
        if len(neigh_tokens) == 0:
            return target_win
        # 2) 沿尺度维堆叠
        tokens = torch.stack([tgt_token] + neigh_tokens, dim=2)  # (B, L, S, C)
        # 3) 线性投影
        proj_tokens = self.proj(tokens)                # (B, L, S, D)
        q = self.q_proj(proj_tokens[:, :, 0, :])       # (B, L, D)
        k = self.k_proj(proj_tokens)                   # (B, L, S, D)
        v = self.v_proj(proj_tokens)                   # (B, L, S, D)
        # 4) 注意力
        D = q.shape[-1]
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / (D ** 0.5)  # (B, L, S)
        attn = torch.softmax(scores, dim=2)                     # (B, L, S)
        fused = (attn.unsqueeze(-1) * v).sum(dim=2)             # (B, L, D)
        # 5) 回映射并注入
        delta = self.out(fused).view(B, L, C, 1, 1)
        return target_win + delta.expand(-1, -1, -1, w, w)


class MS_SAWFFCM_HCFA(nn.Module):

    def __init__(self, in_channels, win_size=8, groups=6, proj_dim=32,
                 enable_intra=True, enable_cross=True, enable_back_inject=True,
                 sa_enable_freq=True, sa_enable_spatial=True, sa_enable_fuse_conv=True, sa_enable_cfse=True):
        super().__init__()
        self.win = win_size
        self.groups = groups
        self.enable_intra = bool(enable_intra)
        self.enable_cross = bool(enable_cross)
        self.enable_back_inject = bool(enable_back_inject)


        if self.enable_intra:
            self.sa_blocks = nn.ModuleList([
                SA_WFFCM(c,
                         win_size=win_size,
                         groups=groups,
                         enable_freq=sa_enable_freq,
                         enable_spatial=sa_enable_spatial,
                         enable_fuse_conv=sa_enable_fuse_conv,
                         enable_cfse=sa_enable_cfse)
                for c in in_channels
            ])
        else:
            self.sa_blocks = None
        self.align = nn.ModuleList([
            nn.Conv2d(c, in_channels[0], 1) if c != in_channels[0] else nn.Identity()
            for c in in_channels
        ])
        self.cross_attn = CrossScaleFreqAttention(in_channels[0], proj_dim=proj_dim)

        self.fuse_convs = nn.ModuleList([nn.Conv2d(in_channels[0], c, 1) for c in in_channels])

    def _extract_windows(self, feat, target_hw):
        B, C, H, W = feat.shape
        Ht, Wt = target_hw
        if (H, W) != (Ht, Wt):
            feat = F.interpolate(feat, size=(Ht, Wt), mode='bilinear', align_corners=False)
        pad_h = (self.win - Ht % self.win) % self.win
        pad_w = (self.win - Wt % self.win) % self.win
        feat = F.pad(feat, (0, pad_w, 0, pad_h))
        unfold = F.unfold(feat, kernel_size=self.win, stride=self.win)
        B_, Cw, L = unfold.shape
        win = unfold.view(B, C, self.win, self.win, L).permute(0, 4, 1, 2, 3)
        return win, (Ht, Wt, pad_h, pad_w)

    def _fold_windows(self, win, out_hw, pad_h, pad_w):
        B, L, C, w, _ = win.shape
        H, W = out_hw
        merged = win.permute(0, 2, 3, 4, 1).reshape(B, C * w * w, L)
        recon = F.fold(merged, output_size=(H + pad_h, W + pad_w), kernel_size=w, stride=w)
        return recon[:, :, :H, :W]

    def forward(self, feats):
        # feats: list [f3, f4, f5]
        if self.enable_intra and self.sa_blocks is not None:
            enhanced = [blk(f) for blk, f in zip(self.sa_blocks, feats)]
        else:
            enhanced = feats

        ref = enhanced[0]
        B, C, H, W = ref.shape

        if self.enable_cross:
            wins = []
            pads = []
            for i, f in enumerate(enhanced):
                fa = self.align[i](f)
                wi, pad = self._extract_windows(fa, (H, W))
                wins.append(wi)
                pads.append(pad)

            updated = self.cross_attn(wins[0], wins[1:])
            Ht, Wt, pad_h, pad_w = pads[0]
            ref_recon = self._fold_windows(updated, (Ht, Wt), pad_h, pad_w)
        else:

            ref_recon = ref


        outs = [self.fuse_convs[0](ref_recon) + feats[0]]
        if self.enable_back_inject:
            for i in range(1, len(feats)):
                tgt = feats[i]
                ref_down = F.interpolate(ref_recon, size=tgt.shape[-2:], mode='bilinear', align_corners=False)
                outs.append(self.fuse_convs[i](ref_down) + tgt)
        else:

            for i in range(1, len(feats)):
                outs.append(enhanced[i])
        return outs


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,
                 enable_ms_enhance=True,  # 是否启用 MS_SAWFFCM_HCFA（整块开关，便于消融）
                 ms_win_size=2, ms_groups=4, ms_proj_dim=32,
                 sa_enable_freq=True, sa_enable_spatial=True, sa_enable_fuse_conv=True, sa_enable_cfse=True,
                 ms_enable_intra=True, ms_enable_cross=True, ms_enable_back_inject=True):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.enable_ms_enhance = bool(enable_ms_enhance)


        in_channels_from_encoder = getattr(self.encoder, 'in_channels', None)
        if self.enable_ms_enhance and isinstance(in_channels_from_encoder, (list, tuple)) and len(in_channels_from_encoder) > 0:
            self._wffcm_built = True
            self.feat_enhancers = MS_SAWFFCM_HCFA(
                in_channels=in_channels_from_encoder,
                win_size=ms_win_size,
                groups=ms_groups,
                proj_dim=ms_proj_dim,
                enable_intra=ms_enable_intra,
                enable_cross=ms_enable_cross,
                enable_back_inject=ms_enable_back_inject,
                sa_enable_freq=sa_enable_freq,
                sa_enable_spatial=sa_enable_spatial,
                sa_enable_fuse_conv=sa_enable_fuse_conv,
                sa_enable_cfse=sa_enable_cfse
            )
        else:
            self._wffcm_built = False
            self.feat_enhancers = None

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        feats = list(feats)


        if self.enable_ms_enhance and not self._wffcm_built:
            self.feat_enhancers = MS_SAWFFCM_HCFA(
                in_channels=[f.shape[1] for f in feats],
                win_size=8,
                groups=4,
                proj_dim=32,
                enable_intra=True, enable_cross=True, enable_back_inject=True
            ).to(feats[0].device)
            self._wffcm_built = True


        enhanced_feats = self.feat_enhancers(feats) if (self.enable_ms_enhance and self.feat_enhancers is not None) else feats


        x = self.encoder(enhanced_feats)
        x = self.decoder(x, targets)
        return x

    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
