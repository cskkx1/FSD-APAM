# # # # 文件路径：ultralytics/nn/modules/iafm.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ultralytics.nn.modules.conv import Conv  # YOLO 原生 Conv

# # ========= WTConv2d 简化版 =========
# # 如果你已经有 WTConv2d 定义，可以删掉这段，用你的那份实现，然后 from ... import WTConv2d 即可。

# import pywt
# from functools import partial

# def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
#     dec_filters = torch.stack([
#         dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
#     ], dim=0)
#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
#     rec_filters = torch.stack([
#         rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
#     ], dim=0)
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
#     return dec_filters, rec_filters

# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x

# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x

# class _ScaleModule(nn.Module):
#     def __init__(self, dims, init_scale=1.0):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

#     def forward(self, x):
#         return self.weight * x

# class WTConv2d(nn.Module):
#     """
#     简化版 WTConv2d：只保留必要内容供 IAFM 使用
#     """
#     def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
#         super().__init__()
#         c = in_channels
#         self.in_channels = c
#         self.wt_levels = wt_levels

#         wt_filter, iwt_filter = create_wavelet_filter(wt_type, c, c, torch.float)
#         self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

#         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
#         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

#         self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", stride=1, groups=c, bias=True)
#         self.base_scale = _ScaleModule([1, c, 1, 1])

#         self.wavelet_convs = nn.ModuleList(
#             nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", stride=1, groups=c * 4, bias=False)
#             for _ in range(wt_levels)
#         )
#         self.wavelet_scale = nn.ModuleList(
#             _ScaleModule([1, c * 4, 1, 1], init_scale=0.1)
#             for _ in range(wt_levels)
#         )

#     def forward(self, x):
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#         curr_x_ll = x

#         # 多尺度小波分解
#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2) or (curr_shape[3] % 2):
#                 curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:, :, 0, :, :]  # 低频 LL

#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

#         next_x_ll = 0
#         # 多尺度小波重建
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll

#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = self.iwt_function(curr_x)
#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         x_tag = next_x_ll
#         # 基础深度卷积
#         x_base = self.base_scale(self.base_conv(x))
#         out = x_base + x_tag
#         return out

# # ========= 对比度增强模块 =========

# class ContrastEnhance(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         return self.max_pool(x) - self.avg_pool(x)

# # ========= IAFM 主模块 =========

# class WTIAFM_POOL(nn.Module):
#     """
#     IAFM-WT：
#     - 局部：3x3 Conv + WTConv2d + MaxPool-AvgPool 对比度
#     - 全局：下采样多头注意力
#     - 融合：通道级门控
#     """
#     def __init__(self, c1, num_heads=4, reduction=4):
#         super().__init__()
#         c = c1
#         self.num_heads = num_heads
#         self.head_dim = c // num_heads
#         assert self.head_dim * num_heads == c, "IAFM: 通道数必须能被 num_heads 整除"

#         # 局部分支
#         self.local_conv3 = Conv(c, c, k=3, s=1)
#         self.local_wt = WTConv2d(c, kernel_size=5, wt_levels=1, wt_type="db1")
#         self.local_contrast = ContrastEnhance(k=3)
#         self.local_fuse = Conv(3 * c, c, k=1, s=1)  # 三路 concat -> C

#         # 下采样注意力
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.q_proj = nn.Linear(c, c)
#         self.k_proj = nn.Linear(c, c)
#         self.v_proj = nn.Linear(c, c)
#         self.attn_out = nn.Linear(c, c)

#         # 通道级门控
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         mid = max(c // reduction, 4)
#         self.gate_fc1 = nn.Linear(2 * c, mid)
#         self.gate_fc2 = nn.Linear(mid, 2 * c)
#         self.gate_act = nn.SiLU(inplace=True)

#         # 输出卷积
#         self.out_conv = Conv(c, c, k=1, s=1)

#     def forward(self, x):
#         if isinstance(x, (list, tuple)):
#             x = x[0]
#         b, c, h, w = x.shape

#         # 局部：Conv + WTConv + Contrast
#         local3 = self.local_conv3(x)
#         local_wt = self.local_wt(x)
#         contrast = self.local_contrast(x)
#         x_local = self.local_fuse(torch.cat([local3, local_wt, contrast], dim=1))

#         # 下采样注意力
#         x_ds = self.pool(x_local)
#         h2, w2 = x_ds.shape[2:]
#         x_seq = x_ds.flatten(2).transpose(1, 2)
#         q = self.q_proj(x_seq)
#         k = self.k_proj(x_seq)
#         v = self.v_proj(x_seq)

#         def reshape_heads(t):
#             B, N, C = t.shape
#             return t.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         q = reshape_heads(q)
#         k = reshape_heads(k)
#         v = reshape_heads(v)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         attn_out = torch.matmul(attn_probs, v)

#         attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(b, h2 * w2, c)
#         attn_out = self.attn_out(attn_out)
#         x_attn_ds = attn_out.transpose(1, 2).view(b, c, h2, w2)
#         x_attn = self.upsample(x_attn_ds)

#         # 通道级门控
#         gap_local = self.gap(x_local).view(b, c)
#         gap_attn = self.gap(x_attn).view(b, c)
#         gate_feat = torch.cat([gap_local, gap_attn], dim=1)
#         gate_mid = self.gate_act(self.gate_fc1(gate_feat))
#         gate_vec = self.gate_fc2(gate_mid)
#         w_local, w_attn = gate_vec.chunk(2, dim=1)
#         w_local = torch.sigmoid(w_local).view(b, c, 1, 1)
#         w_attn = torch.sigmoid(w_attn).view(b, c, 1, 1)

#         x_fused = w_local * x_local + w_attn * x_attn

#         out = self.out_conv(x_fused)
#         return out


# 效果最好的一版
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from functools import partial
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. 小波变换核心工具函数 (保持不变)
# ==========================================
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    # 反转并转换为 Tensor
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    
    # 生成分解滤波器 (LL, LH, HL, HH)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 生成重构滤波器
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return self.weight * x

# ==========================================
# 2. WTConv2d 模块 (保持不变)
# ==========================================
class WTConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
        super().__init__()
        c = in_channels
        self.wt_levels = wt_levels

        # 初始化小波滤波器 (冻结参数)
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, c, c, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 基础卷积分支
        self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", stride=1, groups=c, bias=True)
        self.base_scale = _ScaleModule([1, c, 1, 1])

        # 小波卷积分支 (处理4个频带)
        self.wavelet_convs = nn.ModuleList(
            nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", stride=1, groups=c * 4, bias=False)
            for _ in range(wt_levels)
        )
        self.wavelet_scale = nn.ModuleList(
            _ScaleModule([1, c * 4, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        )

    def forward(self, x):
        # 1. 多级小波分解
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # 处理奇数尺寸 padding
            if (curr_shape[2] % 2) or (curr_shape[3] % 2):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
            
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :] # 取出 LL 继续下一层分解

            # 对当前层频带做卷积增强
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # 2. 多级小波重构
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll # 累加低频
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]] # 裁剪 padding

        x_tag = next_x_ll
        x_base = self.base_scale(self.base_conv(x))
        return x_base + x_tag

# ==========================================
# 3. 对比度增强模块 (Max - Avg)
# ==========================================
class ContrastEnhance(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # k=3 对微小目标最敏感
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        # 简单差分：提取相对于局部背景的高亮/突兀点
        return self.max_pool(x) - self.avg_pool(x)

# ==========================================
# 4. IAFM 主模块 (集成 Softmax 门控)
# ==========================================
class WTIAFM_POOL(nn.Module):
    def __init__(self, c1, num_heads=4, reduction=4):
        super().__init__()
        c = c1
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        
        # --- 局部感知分支 ---
        self.local_conv3 = Conv(c, c, k=3, s=1)
        self.local_wt = WTConv2d(c, kernel_size=5, wt_levels=1, wt_type="db1")
        self.local_contrast = ContrastEnhance(k=3)
        self.local_fuse = Conv(3 * c, c, k=1, s=1) # 融合三种特征

        # --- 全局上下文分支 (轻量化) ---
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 下采样减少计算量并滤波
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.q_proj = nn.Linear(c, c)
        self.k_proj = nn.Linear(c, c)
        self.v_proj = nn.Linear(c, c)
        self.attn_out = nn.Linear(c, c)

        # --- 动态门控融合 ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(c // reduction, 8) # 确保中间层维度不过小
        self.gate_fc1 = nn.Linear(2 * c, mid)
        self.gate_fc2 = nn.Linear(mid, 2 * c) # 输出 2c，对应 local 和 attn 的权重
        self.gate_act = nn.SiLU(inplace=True)

        self.out_conv = Conv(c, c, k=1, s=1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        b, c, h, w = x.shape

        # 1. 局部特征提取
        local3 = self.local_conv3(x)      # 空间纹理
        local_wt = self.local_wt(x)       # 频域边缘
        contrast = self.local_contrast(x) # 显著性突起
        x_local = self.local_fuse(torch.cat([local3, local_wt, contrast], dim=1))

        # 2. 全局注意力计算 (Downsampled)
        x_ds = self.pool(x_local)
        h2, w2 = x_ds.shape[2:]
        x_seq = x_ds.flatten(2).transpose(1, 2) # (B, N, C)

        q = self.q_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_probs = attn_scores.softmax(dim=-1)
        x_attn_out = (attn_probs @ v).transpose(1, 2).reshape(b, -1, c)
        
        x_attn_out = self.attn_out(x_attn_out)
        x_attn_ds = x_attn_out.transpose(1, 2).view(b, c, h2, w2)
        x_attn = self.upsample(x_attn_ds)

        # 3. Softmax 门控融合 (Optimization Here!)
        gap_local = self.gap(x_local).view(b, c)
        gap_attn = self.gap(x_attn).view(b, c)
        
        gate_feat = torch.cat([gap_local, gap_attn], dim=1) # (B, 2C)
        gate_mid = self.gate_act(self.gate_fc1(gate_feat))
        gate_vec = self.gate_fc2(gate_mid) # (B, 2C)

        # 将权重 reshape 为 (B, C, 2) 并做 Softmax
        gate_weights = F.softmax(gate_vec.view(b, c, 2), dim=2) 
        w_local = gate_weights[:, :, 0].view(b, c, 1, 1)
        w_attn = gate_weights[:, :, 1].view(b, c, 1, 1)

        # 加权融合
        x_fused = w_local * x_local + w_attn * x_attn

        return self.out_conv(x_fused)


# 改进上后
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pywt
# import math

# # ==========================================
# # 工具函数：生成小波滤波器
# # ==========================================
# def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
#     w = pywt.Wavelet(wave)
#     # 反转并转换为 Tensor
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    
#     # 生成分解滤波器 (LL, LH, HL, HH)
#     dec_filters = torch.stack([
#         dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
#     ], dim=0)
    
#     # (4, 1, k, k) -> (4*in_channels, 1, k, k) 适配 groups=in_channels
#     # 注意：这里为了适配 Conv2d 的 group 操作，我们需要对 channel 进行维度的调整
#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

#     # 生成重构滤波器
#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
#     rec_filters = torch.stack([
#         rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
#     ], dim=0)
    
#     # (4, 1, k, k) -> (4*out_channels, 1, k, k)
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    
#     return dec_filters, rec_filters

# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     # 使用 groups=c 进行逐通道卷积，实现分解
#     # 输出 shape: (b, 4*c, h/2, w/2)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     # Reshape 为 (b, c, 4, h/2, w/2) 方便后续分离 LL, LH, HL, HH
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x

# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     # Reshape 回 (b, 4*c, h/2, w/2) 准备进行转置卷积重构
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x

# # ==========================================
# # [创新点组件] 频带重加权模块 (原 _ScaleModule)
# # 对应论文：2.1 节 "可学习的频带重加权策略"
# # ==========================================
# class FrequencyBandReweighting(nn.Module):
#     def __init__(self, dims, init_scale=0.1):
#         super().__init__()
#         # 初始化较小的值，使得初始阶段主要依赖 Base Conv，
#         # 随着训练进行，网络自动学习需要放大哪些高频细节。
#         self.scale = nn.Parameter(torch.ones(*dims) * init_scale)

#     def forward(self, x):
#         return x * self.scale

# # ==========================================
# # 2. WTConv2d 模块 (核心改进版)
# # 对应论文：2.1 节 "基于频空联合解耦的细节补偿机制"
# # ==========================================
# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
#         super().__init__()
#         c = in_channels
#         self.wt_levels = wt_levels
#         self.wt_type = wt_type

#         # 1. 初始化小波滤波器 (冻结参数，作为固定算子)
#         # 使用 'db1' (Haar) 是为了对工业场景的阶跃型瑕疵（Step-change）最敏感
#         wt_filter, iwt_filter = create_wavelet_filter(wt_type, c, c, torch.float)
#         self.register_buffer('wt_filter', wt_filter)
#         self.register_buffer('iwt_filter', iwt_filter)

#         # 2. 空间主干分支 (保持语义连贯性)
#         self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", stride=1, groups=c, bias=True)
#         self.base_scale = FrequencyBandReweighting([1, c, 1, 1], init_scale=1.0) # 主干默认权重为1

#         # 3. 频域细节分支 (处理4个频带: LL, LH, HL, HH)
#         self.wavelet_convs = nn.ModuleList(
#             nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", stride=1, groups=c * 4, bias=False)
#             for _ in range(wt_levels)
#         )
        
#         # [创新点体现] 可学习的频带重加权
#         # 允许网络自动抑制噪声频带，增强有效纹理频带
#         self.freq_reweighting = nn.ModuleList(
#             FrequencyBandReweighting([1, c * 4, 1, 1], init_scale=0.1)
#             for _ in range(wt_levels)
#         )

#     def forward(self, x):
#         # ---------------------------
#         # Step 1: 多级小波分解 (Analysis Path)
#         # ---------------------------
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
            
#             # 处理奇数尺寸 padding，保证小波变换尺寸对齐
#             if (curr_shape[2] % 2) or (curr_shape[3] % 2):
#                 curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

#             # 执行小波变换 (Forward DWT)
#             curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            
#             # 分离低频 (LL) 用于下一级，保存高频 (LH, HL, HH)
#             curr_x_ll = curr_x[:, :, 0, :, :] 
            
#             # 准备进行频带处理
#             shape_x = curr_x.shape
#             # Flatten 频带维度: (b, c, 4, h, w) -> (b, c*4, h, w)
#             # 这样可以用 group conv 独立处理每个频带
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            
#             # --- 频域处理核心 ---
#             # 1. 卷积处理细节
#             curr_x_tag = self.wavelet_convs[i](curr_x_tag)
#             # 2. [创新点] 频带重加权 (Frequency Attention)
#             curr_x_tag = self.freq_reweighting[i](curr_x_tag)
#             # --------------------

#             # 还原形状
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

#         # ---------------------------
#         # Step 2: 多级小波重构 (Synthesis Path)
#         # ---------------------------
#         next_x_ll = 0 
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             # 将上一层的低频加回来
#             curr_x_ll = curr_x_ll + next_x_ll 
            
#             # 拼接 LL 和 High Frequency Components
#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

#             # 执行逆小波变换 (Inverse DWT)
#             next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            
#             # 裁剪掉之前为了对齐而 pad 的部分
#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         # ---------------------------
#         # Step 3: 残差融合 (Residual Fusion)
#         # ---------------------------
#         x_tag = next_x_ll
#         # 主干分支：标准卷积
#         x_base = self.base_scale(self.base_conv(x))
        
#         # 返回：空间主干 + 频域高频细节补偿
#         return x_base + x_tag

# # ==========================================
# # 3. 对比度增强模块 (Contrast)
# # 对应论文：2.2 节 "空域显著性解耦"
# # ==========================================
# class SpatialSaliencyDecoupling(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         # 显式命名为 Saliency Decoupling 体现论文逻辑
#         # k=3 模拟局部感受野，差分操作模拟人眼对突兀点的捕捉
#         self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         # 核心公式：Salience = Max(local) - Avg(local)
#         # 这是一种非线性的高通滤波，专门提取 "异常亮点"
#         return self.max_pool(x) - self.avg_pool(x)

# # ==========================================
# # 4. IAFM 主模块
# # 对应论文：2.3 节 "全局上下文建模与门控聚合"
# # ==========================================
# class WTIAFM_POOL(nn.Module):
#     def __init__(self, c1, num_heads=4, reduction=4):
#         super().__init__()
#         c = c1
#         self.num_heads = num_heads
#         self.head_dim = c // num_heads

#         # --- 分支1: 局部细节感知 (Local Perception) ---
#         self.local_conv3 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#         # 频域细节解耦 (创新点1)
#         self.local_wt = WTConv2d(c, kernel_size=5, wt_levels=1, wt_type="db1")
#         # 空域显著性解耦 (创新点2)
#         self.local_saliency = SpatialSaliencyDecoupling(k=3)
        
#         self.local_fuse = nn.Sequential(
#             nn.Conv2d(3 * c, c, kernel_size=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )

#         # --- 分支2: 全局上下文建模 (Global Context) ---
#         # 轻量化 Self-Attention
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.q_proj = nn.Linear(c, c)
#         self.k_proj = nn.Linear(c, c)
#         self.v_proj = nn.Linear(c, c)
#         self.attn_out = nn.Linear(c, c)

#         # --- 模块3: 动态门控融合 (Gated Aggregation) ---
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         mid = max(c // reduction, 8)
#         self.gate_fc1 = nn.Linear(2 * c, mid)
#         self.gate_fc2 = nn.Linear(mid, 2 * c)
#         self.gate_act = nn.SiLU(inplace=True)

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape

#         # 1. 局部特征提取 (Multi-view Local Features)
#         local3 = self.local_conv3(x)           # 语义视图
#         local_wt = self.local_wt(x)            # 频域视图 (Edges)
#         contrast = self.local_saliency(x)      # 显著性视图 (Saliency)
        
#         # 融合三种局部视图
#         x_local = self.local_fuse(torch.cat([local3, local_wt, contrast], dim=1))

#         # 2. 全局注意力计算 (Global Context)
#         x_ds = self.pool(x_local)
#         b, c, h2, w2 = x_ds.shape
#         x_seq = x_ds.flatten(2).transpose(1, 2)

#         q = self.q_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
#         attn_probs = attn_scores.softmax(dim=-1)
#         x_attn_out = (attn_probs @ v).transpose(1, 2).reshape(b, -1, c)

#         x_attn_out = self.attn_out(x_attn_out)
#         x_attn_ds = x_attn_out.transpose(1, 2).view(b, c, h2, w2)
        
#         # 上采样回原尺寸
#         x_attn = F.interpolate(x_attn_ds, size=(h, w), mode="bilinear", align_corners=False)

#         # 3. 动态门控融合 (Adaptive Gating)
#         gap_local = self.gap(x_local).view(b, c)
#         gap_attn = self.gap(x_attn).view(b, c)

#         gate_feat = torch.cat([gap_local, gap_attn], dim=1)
#         gate_mid = self.gate_act(self.gate_fc1(gate_feat))
#         gate_vec = self.gate_fc2(gate_mid)

#         # 生成互斥的软门控权重: w_local + w_attn = 1
#         gate_weights = F.softmax(gate_vec.view(b, c, 2), dim=2)
#         w_local = gate_weights[:, :, 0].view(b, c, 1, 1)
#         w_attn = gate_weights[:, :, 1].view(b, c, 1, 1)

#         # 加权融合
#         x_fused = w_local * x_local + w_attn * x_attn

#         return self.out_conv(x_fused)

# # 测试代码
# if __name__ == "__main__":
#     x = torch.randn(2, 64, 64, 64) # Batch=2, Channel=64, H=64, W=64
#     model = IAFM(c1=64)
#     out = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out.shape}")


# 改进上中后
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pywt

# # ==========================================
# # 工具函数：生成小波滤波器
# # ==========================================
# def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    
#     dec_filters = torch.stack([
#         dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
#     ], dim=0)
#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
#     rec_filters = torch.stack([
#         rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
#     ], dim=0)
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    
#     return dec_filters, rec_filters

# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x

# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x

# # ==========================================
# # 频带重加权模块
# # ==========================================
# class FrequencyBandReweighting(nn.Module):
#     def __init__(self, dims, init_scale=0.1):
#         super().__init__()
#         self.scale = nn.Parameter(torch.ones(*dims) * init_scale)

#     def forward(self, x):
#         return x * self.scale

# # ==========================================
# # WTConv2d 模块 (频域细节解耦单元 FDD)
# # ==========================================
# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
#         super().__init__()
#         c = in_channels
#         self.wt_levels = wt_levels
#         self.wt_type = wt_type

#         wt_filter, iwt_filter = create_wavelet_filter(wt_type, c, c, torch.float)
#         self.register_buffer('wt_filter', wt_filter)
#         self.register_buffer('iwt_filter', iwt_filter)

#         self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", stride=1, groups=c, bias=True)
#         self.base_scale = FrequencyBandReweighting([1, c, 1, 1], init_scale=1.0)

#         self.wavelet_convs = nn.ModuleList(
#             nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", stride=1, groups=c * 4, bias=False)
#             for _ in range(wt_levels)
#         )
        
#         self.freq_reweighting = nn.ModuleList(
#             FrequencyBandReweighting([1, c * 4, 1, 1], init_scale=0.1)
#             for _ in range(wt_levels)
#         )

#     def forward(self, x):
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
            
#             if (curr_shape[2] % 2) or (curr_shape[3] % 2):
#                 curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

#             curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
#             curr_x_ll = curr_x[:, :, 0, :, :] 
            
#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_convs[i](curr_x_tag)
#             curr_x_tag = self.freq_reweighting[i](curr_x_tag)
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

#         next_x_ll = 0 
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll 
#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         x_tag = next_x_ll
#         x_base = self.base_scale(self.base_conv(x))
#         return x_base + x_tag

# # ==========================================
# # 空域显著性解耦单元 (SSD) - 双路并行版
# # ==========================================
# class SpatialSaliencyDecoupling(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         # 路径 1: Max-Avg 对比度增强
#         self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
        
#         # 路径 2: Center-Surround 侧抑制
#         # 使用空洞卷积或者自定义核来实现"挖空中心"的效果
#         # 这里简化为：用较大的 Avg 代表"周围"，用原图代表"中心"
#         self.surround_pool = nn.AvgPool2d(kernel_size=k+2, stride=1, padding=(k+2) // 2)
        
#     def forward(self, x):
#         # 路径 1: Contrast (Max - Avg)
#         contrast_map = self.max_pool(x) - self.avg_pool(x)
        
#         # 路径 2: Isolation (Center - Surround)
#         # Center 用原始或小池化，Surround 用大池化
#         isolation_map = x - self.surround_pool(x)
        
#         # 双重确认：两者相乘
#         # 只有"对比度高"且"孤立"的点才会被保留
#         saliency = contrast_map * torch.sigmoid(isolation_map)
        
#         return saliency

# # ==========================================
# # FSD-APAM 主模块
# # ==========================================
# class WTIAFM_POOL(nn.Module):
#     def __init__(self, c1, num_heads=4, reduction=4):
#         super().__init__()
#         c = c1
#         self.num_heads = num_heads
#         self.head_dim = c // num_heads

#         # --- 分支1: 局部细节感知 ---
#         self.local_conv3 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#         # 频域细节解耦 FDD
#         self.local_wt = WTConv2d(c, kernel_size=5, wt_levels=1, wt_type="db1")
#         # 空域显著性解耦 SSD (双路并行版)
#         self.local_saliency = SpatialSaliencyDecoupling(k=3)
        
#         self.local_fuse = nn.Sequential(
#             nn.Conv2d(3 * c, c, kernel_size=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )

#         # --- 分支2: 全局上下文建模 (暂不改动) ---
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.q_proj = nn.Linear(c, c)
#         self.k_proj = nn.Linear(c, c)
#         self.v_proj = nn.Linear(c, c)
#         self.attn_out = nn.Linear(c, c)

#         # --- 模块3: 动态门控融合 ---
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         mid = max(c // reduction, 8)
#         self.gate_fc1 = nn.Linear(2 * c, mid)
#         self.gate_fc2 = nn.Linear(mid, 2 * c)
#         self.gate_act = nn.SiLU(inplace=True)

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape

#         # 1. 局部特征提取 (三种视图)
#         local3 = self.local_conv3(x)
#         local_wt = self.local_wt(x)
#         contrast = self.local_saliency(x)
        
#         x_local = self.local_fuse(torch.cat([local3, local_wt, contrast], dim=1))

#         # 2. 全局注意力计算
#         x_ds = self.pool(x_local)
#         b, c, h2, w2 = x_ds.shape
#         x_seq = x_ds.flatten(2).transpose(1, 2)

#         q = self.q_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x_seq).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
#         attn_probs = attn_scores.softmax(dim=-1)
#         x_attn_out = (attn_probs @ v).transpose(1, 2).reshape(b, -1, c)

#         x_attn_out = self.attn_out(x_attn_out)
#         x_attn_ds = x_attn_out.transpose(1, 2).view(b, c, h2, w2)
#         x_attn = F.interpolate(x_attn_ds, size=(h, w), mode="bilinear", align_corners=False)

#         # 3. Softmax 门控融合
#         gap_local = self.gap(x_local).view(b, c)
#         gap_attn = self.gap(x_attn).view(b, c)

#         gate_feat = torch.cat([gap_local, gap_attn], dim=1)
#         gate_mid = self.gate_act(self.gate_fc1(gate_feat))
#         gate_vec = self.gate_fc2(gate_mid)

#         gate_weights = F.softmax(gate_vec.view(b, c, 2), dim=2)
#         w_local = gate_weights[:, :, 0].view(b, c, 1, 1)
#         w_attn = gate_weights[:, :, 1].view(b, c, 1, 1)

#         x_fused = w_local * x_local + w_attn * x_attn

#         return self.out_conv(x_fused)

# # 测试代码
# if __name__ == "__main__":
#     x = torch.randn(2, 64, 64, 64)
#     model = FSDAPAM(c1=64)
#     out = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out.shape}")


# # 改进上中下后
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pywt

# # =========================================================================
# # 1. 基础工具：小波滤波器生成与变换
# # =========================================================================
# def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    
#     dec_filters = torch.stack([
#         dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#         dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
#     ], dim=0)
#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
#     rec_filters = torch.stack([
#         rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#         rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
#     ], dim=0)
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    
#     return dec_filters, rec_filters

# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x

# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x

# # =========================================================================
# # 2. 核心模块一：频域细节解耦 (FDD) - WTConv2d with Reweighting
# # =========================================================================
# class FrequencyBandReweighting(nn.Module):
#     """ 可学习的频带重加权模块 """
#     def __init__(self, dims, init_scale=0.1):
#         super().__init__()
#         # 初始化较小，允许网络自动学习增强关键频带
#         self.scale = nn.Parameter(torch.ones(*dims) * init_scale)

#     def forward(self, x):
#         return x * self.scale

# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
#         super().__init__()
#         c = in_channels
#         self.wt_levels = wt_levels

#         # 初始化小波滤波器 (Haar/db1 对阶跃信号最敏感)
#         wt_filter, iwt_filter = create_wavelet_filter(wt_type, c, c, torch.float)
#         self.register_buffer('wt_filter', wt_filter)
#         self.register_buffer('iwt_filter', iwt_filter)

#         # 空间主干分支
#         self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", stride=1, groups=c, bias=True)
#         self.base_scale = FrequencyBandReweighting([1, c, 1, 1], init_scale=1.0)

#         # 频域卷积分支
#         self.wavelet_convs = nn.ModuleList(
#             nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", stride=1, groups=c * 4, bias=False)
#             for _ in range(wt_levels)
#         )
        
#         # [创新点] 频带重加权
#         self.freq_reweighting = nn.ModuleList(
#             FrequencyBandReweighting([1, c * 4, 1, 1], init_scale=0.1)
#             for _ in range(wt_levels)
#         )

#     def forward(self, x):
#         # 1. 小波分解
#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []
#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2) or (curr_shape[3] % 2):
#                 curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

#             curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
#             curr_x_ll = curr_x[:, :, 0, :, :] 
            
#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            
#             # 频域增强处理
#             curr_x_tag = self.wavelet_convs[i](curr_x_tag)
#             curr_x_tag = self.freq_reweighting[i](curr_x_tag) # Reweighting
            
#             curr_x_tag = curr_x_tag.reshape(shape_x)
#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

#         # 2. 小波重构
#         next_x_ll = 0 
#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll 
#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         # 3. 残差融合
#         x_tag = next_x_ll
#         x_base = self.base_scale(self.base_conv(x))
#         return x_base + x_tag

# # =========================================================================
# # 3. 核心模块二：空域显著性解耦 (SSD) - 双路并行机制
# # =========================================================================
# class SpatialSaliencyDecoupling(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         # 路径 A: Max-Avg (局部对比度增强)
#         self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
        
#         # 路径 B: Center-Surround (侧抑制/邻域抑制)
#         # 模拟生物视觉：中心(x) - 周围(大核Avg)
#         self.surround_pool = nn.AvgPool2d(kernel_size=k+2, stride=1, padding=(k+2) // 2)
        
#     def forward(self, x):
#         # 1. 计算局部对比度 (突兀程度)
#         contrast_map = self.max_pool(x) - self.avg_pool(x)
        
#         # 2. 计算邻域抑制 (孤立程度)
#         isolation_map = x - self.surround_pool(x)
        
#         # 3. 双重确认：只有既突兀又孤立的点，才被认为是微小缺陷
#         # 使用 sigmoid 将 isolation 映射为 0-1 的门控权重
#         saliency = contrast_map * torch.sigmoid(isolation_map)
        
#         return saliency

# # =========================================================================
# # 4. 核心模块三：自适应感知聚合 (APA) - 稀疏-密集双流交互注意力
# # =========================================================================
# class SparseDenseContext(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         self.channels = channels
#         mid_c = channels // reduction
        
#         # --- 密集流 (Dense Path): 网格化背景语义 ---
#         # 将特征图划分为大网格 (Grid)，提取粗粒度语义
#         self.grid_pool = nn.AdaptiveAvgPool2d((4, 4)) # 变成 4x4 的 grid
#         self.dense_mlp = nn.Sequential(
#             nn.Linear(channels, mid_c),
#             nn.LayerNorm(mid_c),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid_c, channels)
#         )
        
#         # --- 稀疏流 (Sparse Path): 关键点交互 ---
#         # 仅关注 Top-K 个高响应点 (Top-K Anchors)
#         self.topk_ratio = 0.05 # 仅选取前 5% 的点
#         self.sparse_mlp = nn.Sequential(
#             nn.Linear(channels, mid_c),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid_c, channels)
#         )
        
#         self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, h, w = x.shape
        
#         # 1. Dense Path (Grid Context)
#         # (B, C, 4, 4) -> (B, 16, C)
#         x_dense = self.grid_pool(x).flatten(2).transpose(1, 2)
#         x_dense = self.dense_mlp(x_dense) 
#         # 插值回原尺寸 (B, C, H, W)
#         x_dense = x_dense.transpose(1, 2).reshape(b, c, 4, 4)
#         x_dense = F.interpolate(x_dense, size=(h, w), mode='bilinear', align_corners=False)
        
#         # 2. Sparse Path (Anchor Context)
#         # 基于幅值选择 Top-K 点
#         x_flat = x.flatten(2).transpose(1, 2) # (B, H*W, C)
#         energy = torch.mean(x_flat, dim=2) # (B, H*W)
        
#         k = int(h * w * self.topk_ratio)
#         # 获取 Top-K 索引
#         topk_val, topk_idx = torch.topk(energy, k, dim=1) 
        
#         # 聚集稀疏特征 (B, K, C)
#         batch_idx = torch.arange(b, device=x.device).unsqueeze(1).expand(-1, k)
#         x_sparse_feat = x_flat[batch_idx, topk_idx]
        
#         # 稀疏点交互
#         x_sparse_feat = self.sparse_mlp(x_sparse_feat)
        
#         # 散布回原图 (Scatter back)
#         x_sparse = torch.zeros_like(x_flat)
#         x_sparse[batch_idx, topk_idx] = x_sparse_feat
#         x_sparse = x_sparse.transpose(1, 2).reshape(b, c, h, w)
        
#         # 3. 双流交互融合
#         x_out = self.fusion(torch.cat([x_dense, x_sparse], dim=1))
        
#         return x * self.sigmoid(x_out) # 门控机制注入上下文

# # =========================================================================
# # 5. FSD-APAM 整体组装
# # =========================================================================
# class WTIAFM_POOL(nn.Module):
#     def __init__(self, c1, num_heads=4, reduction=4):
#         super().__init__()
#         c = c1
        
#         # --- 分支1: 局部细节感知 (Local View) ---
#         self.local_conv3 = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )
#         # FDD Unit (频域)
#         self.local_wt = WTConv2d(c, kernel_size=5, wt_levels=1, wt_type="db1")
#         # SSD Unit (空域 - 双路并行)
#         self.local_saliency = SpatialSaliencyDecoupling(k=3)
        
#         self.local_fuse = nn.Sequential(
#             nn.Conv2d(3 * c, c, kernel_size=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )

#         # --- 分支2: 全局上下文建模 (Global View) ---
#         # APA Unit (升级为 Sparse-Dense Context)
#         self.global_context = SparseDenseContext(c)

#         # --- 分支3: 动态门控融合 (Adaptive Gating) ---
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         mid = max(c // reduction, 8)
#         self.gate_fc1 = nn.Linear(2 * c, mid)
#         self.gate_fc2 = nn.Linear(mid, 2 * c)
#         self.gate_act = nn.SiLU(inplace=True)

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1),
#             nn.BatchNorm2d(c),
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape

#         # 1. 局部特征提取 (Local View Extraction)
#         # 同时提取: 语义(Conv), 边缘(WT), 显著点(Saliency)
#         local3 = self.local_conv3(x)
#         local_wt = self.local_wt(x)
#         contrast = self.local_saliency(x)
        
#         # 融合三种局部视图
#         x_local = self.local_fuse(torch.cat([local3, local_wt, contrast], dim=1))

#         # 2. 全局上下文建模 (Global View Modeling)
#         # 使用稀疏-密集双流交互
#         x_global = self.global_context(x_local)

#         # 3. 动态门控融合 (Adaptive Gating)
#         gap_local = self.gap(x_local).view(b, c)
#         gap_global = self.gap(x_global).view(b, c)

#         gate_feat = torch.cat([gap_local, gap_global], dim=1)
#         gate_mid = self.gate_act(self.gate_fc1(gate_feat))
#         gate_vec = self.gate_fc2(gate_mid)

#         # 生成互斥权重 w_local + w_global = 1
#         gate_weights = F.softmax(gate_vec.view(b, c, 2), dim=2)
#         w_local = gate_weights[:, :, 0].view(b, c, 1, 1)
#         w_global = gate_weights[:, :, 1].view(b, c, 1, 1)

#         # 最终加权
#         x_fused = w_local * x_local + w_global * x_global

#         return self.out_conv(x_fused)

# # 测试代码
# if __name__ == "__main__":
#     # 模拟输入: Batch=2, Channel=64, H=64, W=64
#     x = torch.randn(2, 64, 64, 64) 
#     model = FSDAPAM(c1=64)
#     out = model(x)
#     print(f"Model Structure Initialized.")
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out.shape}")




