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
    """
    Scale-Aware Windowed Fused Fourier Convolution Mixer (with local window FFT)
    说明：
    - 即插即用的特征增强模块，输入输出尺寸保持一致 (B, C, H, W)
    - 在空间分支与频域分支上分别提取表征，再进行融合与通道注意力自适应
    - 采用窗口化 rFFT2/irFFT2 以降低频域计算开销（局部窗口内进行频域变换）
    参数：
    - in_ch: 输入特征通道数（输出与输入通道一致）
    - mid_ch: 中间通道数（若为 None 则等于 in_ch）
    - win_size: 频域分支的窗口大小（建议为 8/16/32 等能被特征尺寸较好整除的值）
    - groups: 频域分组（按通道分组，减小 1x1 point-wise 卷积计算量）
    - enable_freq: 是否启用频域分支（可关闭用于消融）
    """
    def __init__(self, in_ch, mid_ch=None, win_size=16, groups=4, enable_freq=True):
        super().__init__()
        mid_ch = mid_ch or in_ch
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.win = int(win_size)
        self.groups = int(groups)
        self.enable_freq = bool(enable_freq)

        # 1) 通道降维：降低后续计算量（维持信息瓶颈）
        self.reduce = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.act = nn.GELU()

        # 2) 空间分支：深度可分离卷积提取空间上下文
        self.spatial = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, groups=mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 1),
            nn.BatchNorm2d(mid_ch),
        )

        # 3) 频域分支（可选）：窗口化 rFFT2 -> 频域 PWConv -> irFFT2
        if enable_freq:
            # groups 上限为 mid_ch，且至少为 1
            self.groups = max(1, min(groups, mid_ch))
            self.freq_pw = nn.ModuleList([
                # 拼接实部/虚部后通道翻倍，因此输入维度为 (group_ch * 2)
                # 为了在 iFFT 前可均分为实部/虚部，这里输出通道需为 2 * group_ch
                nn.Conv2d((mid_ch // self.groups) * 2, (mid_ch // self.groups) * 2, 1)
                for _ in range(self.groups)
            ])
            self.freq_bn = nn.BatchNorm2d(mid_ch)
        else:
            self.freq_pw = None

        # 4) 融合层：跨分支的逐点融合
        self.cross_pw = nn.Conv2d(mid_ch, mid_ch, 1)
        self.cross_bn = nn.BatchNorm2d(mid_ch)

        # 5) 通道注意力（CF-SE）：全局上下文重标定
        self.cf_fc1 = nn.Linear(mid_ch, max(4, mid_ch // 4))
        self.cf_fc2 = nn.Linear(max(4, mid_ch // 4), mid_ch)

        # 6) 输出回写到原通道数，并与残差相加
        self.output = nn.Conv2d(mid_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        mid = self.reduce(x)

        # --- Spatial branch ---
        s = self.spatial(mid)

        # --- Frequency branch with window partition ---
        if not self.enable_freq:
            # 关闭频域分支时，直接置零（仅保留空间分支信息）
            f_recon = torch.zeros_like(s)
        else:
            win = self.win
            stride = win  # 采用非重叠窗口

            # 为保证窗口整除，进行右下方向的 padding，再恢复
            pad_h = (win - H % win) % win
            pad_w = (win - W % win) % win
            mid_p = F.pad(mid, (0, pad_w, 0, pad_h))

            # unfold: B, C*win*win, L  (L = 窗口个数)
            unfold = F.unfold(mid_p, kernel_size=win, stride=stride)
            B_, Cw, L = unfold.shape
            patches = unfold.view(B, self.mid_ch, win, win, L).permute(0, 4, 1, 2, 3)
            patches = patches.reshape(B * L, self.mid_ch, win, win)

            group_ch = self.mid_ch // self.groups
            out_groups = []

            for g in range(self.groups):
                start, end = g * group_ch, (g + 1) * group_ch
                wg = patches[:, start:end, :, :]  # (B*L, gch, win, win)
                # rFFT2：频域表示（正交归一化）
                Fcomp = torch.fft.rfft2(wg, norm='ortho')
                Fr, Fi = Fcomp.real, Fcomp.imag
                Fcat = torch.cat([Fr, Fi], dim=1)
                # 频域逐点卷积（通道混合）
                Fproc = self.freq_pw[g](Fcat)
                # 将输出再拆为实部/虚部（平均切分）
                half = Fproc.shape[1] // 2
                # 当 group_ch 为奇数时，half 取整向下，剩余通道将被忽略（对齐实现）
                Fr2 = Fproc[:, :half, :, :]
                Fi2 = Fproc[:, half:2*half, :, :]
                Frec = torch.complex(Fr2, Fi2)
                # iFFT2：恢复到时域
                rec = torch.fft.irfft2(Frec, s=(win, win), norm='ortho')
                out_groups.append(rec)

            # 融合各组并 fold 回整图
            freq_patches = torch.cat(out_groups, dim=1)  # (B*L, mid_ch, win, win)
            freq_patches = freq_patches.view(B, L, self.mid_ch, win, win)
            freq_patches = freq_patches.permute(0, 2, 3, 4, 1).reshape(B, self.mid_ch * win * win, L)
            f_recon = F.fold(freq_patches, output_size=(H + pad_h, W + pad_w),
                             kernel_size=win, stride=stride)
            f_recon = f_recon[:, :, :H, :W]  # 裁剪回原尺寸
            f_recon = self.freq_bn(f_recon)

        # --- Fusion & CF-SE ---
        fused = s + f_recon
        fused = self.cross_bn(self.cross_pw(fused))
        fused = self.act(fused)
        gap = fused.mean(dim=(2, 3))          # 全局平均池化 (B, C)
        se = self.act(self.cf_fc1(gap))
        se = torch.sigmoid(self.cf_fc2(se)).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        out = fused * se

        out = self.output(out)
        return x + out  # 残差连接，稳定训练

@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
        # ---------------------------------------------------------------------
        # 在骨干后进行各尺度特征增强：为 encoder 的每个输入尺度构建一个 SA_WFFCM
        # 说明：
        # - 优先从 encoder.in_channels 推断每个尺度的通道数，以便在 __init__ 中静态构建模块
        # - 若外部 encoder 无该属性，则在首次 forward 根据实际特征动态构建（懒构建）
        # - win_size / groups 可根据需要调参；默认对中/高分辨率采用 16/4 的设置
        # ---------------------------------------------------------------------
        in_channels_from_encoder = getattr(self.encoder, 'in_channels', None)
        if isinstance(in_channels_from_encoder, (list, tuple)) and len(in_channels_from_encoder) > 0:
            self._wffcm_built = True
            self.feat_enhancers = nn.ModuleList([
                SA_WFFCM(in_ch=c, win_size=16, groups=4, enable_freq=True) for c in in_channels_from_encoder
            ])
        else:
            # 延迟构建：在第一次拿到骨干输出特征后再根据通道数创建
            self._wffcm_built = False
            self.feat_enhancers = nn.ModuleList()
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        # ---------------------------------------------------------------------
        # 1) 骨干提取多尺度特征（通常为 [P3@1/8, P4@1/16, P5@1/32]）
        #    说明：当前仓库配置（如 rtdetr_r50vd.yml）已包含 1/8 尺度（return_idx: [1,2,3]）
        #    若使用其他骨干，请确保返回中包含 1/8 尺度，或在骨干配置中开启相应输出。
        # ---------------------------------------------------------------------
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        feats = list(feats)

        # ---------------------------------------------------------------------
        # 2) 如未在 __init__ 中静态构建增强模块，则在此进行懒构建
        #    注意：为避免设备不一致，构建后将模块迁移到与特征相同的 device
        # ---------------------------------------------------------------------
        if not self._wffcm_built:
            self.feat_enhancers = nn.ModuleList([
                SA_WFFCM(in_ch=f.shape[1], win_size=16, groups=min(4, f.shape[1]), enable_freq=True)
                for f in feats
            ])
            # 将新建模块迁移到当前特征所在设备
            device = feats[0].device
            self.feat_enhancers.to(device)
            self._wffcm_built = True

        # ---------------------------------------------------------------------
        # 3) 尺度特征增强（即插即用）：逐尺度送入 SA_WFFCM，输出与输入形状一致
        # ---------------------------------------------------------------------
        enhanced_feats = [m(f) for m, f in zip(self.feat_enhancers, feats)]

        # ---------------------------------------------------------------------
        # 4) 编码器（FPN/PAN/Transformer Encoder 等）
        # ---------------------------------------------------------------------
        x = self.encoder(enhanced_feats)
        # ---------------------------------------------------------------------
        # 5) 解码器（Transformer Decoder 等）
        # ---------------------------------------------------------------------
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
