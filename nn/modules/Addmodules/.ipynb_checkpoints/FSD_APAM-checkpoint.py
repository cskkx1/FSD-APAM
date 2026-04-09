import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# =========================================================================
# 1. Utilities: Wavelet Filter Generation and Transforms
# =========================================================================
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

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

# =========================================================================
# 2. Core Module 1: Frequency-domain Detail Decoupling (FDD)
# =========================================================================
class FrequencyBandReweighting(nn.Module):
    def __init__(self, dims, init_scale=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return x * self.scale

class FDD(nn.Module):
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
        super().__init__()
        c = in_channels
        self.wt_levels = wt_levels

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, c, c, torch.float)
        self.register_buffer('wt_filter', wt_filter)
        self.register_buffer('iwt_filter', iwt_filter)

        self.base_conv = nn.Conv2d(c, c, kernel_size, padding="same", stride=1, groups=c, bias=True)
        self.base_scale = FrequencyBandReweighting([1, c, 1, 1], init_scale=1.0)

        self.wavelet_convs = nn.ModuleList(
            nn.Conv2d(c * 4, c * 4, kernel_size, padding="same", stride=1, groups=c * 4, bias=False)
            for _ in range(wt_levels)
        )
        
        self.freq_reweighting = nn.ModuleList(
            FrequencyBandReweighting([1, c * 4, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2) or (curr_shape[3] % 2):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :] 
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            
            curr_x_tag = self.wavelet_convs[i](curr_x_tag)
            curr_x_tag = self.freq_reweighting[i](curr_x_tag)
            
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0 
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll 
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        x_base = self.base_scale(self.base_conv(x))
        return x_base + x_tag

# =========================================================================
# 3. Core Module 2: Spatial Salience Decoupling (SSD)
# =========================================================================
class SSD(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.surround_pool = nn.AvgPool2d(kernel_size=k+2, stride=1, padding=(k+2) // 2)
        
    def forward(self, x):
        contrast_map = self.max_pool(x) - self.avg_pool(x)
        isolation_map = x - self.surround_pool(x)
        saliency = contrast_map * torch.sigmoid(isolation_map)
        return saliency

# =========================================================================
# 4. Core Module 3: Adaptive Contextual Perceptual Aggregation (ACPA)
# =========================================================================
class ACPA(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        mid_c = channels // reduction
        
        self.grid_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dense_mlp = nn.Sequential(
            nn.Linear(channels, mid_c),
            nn.LayerNorm(mid_c),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, channels)
        )
        
        self.topk_ratio = 0.05
        self.sparse_mlp = nn.Sequential(
            nn.Linear(channels, mid_c),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, channels)
        )
        
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        
        x_dense = self.grid_pool(x).flatten(2).transpose(1, 2)
        x_dense = self.dense_mlp(x_dense) 
        x_dense = x_dense.transpose(1, 2).reshape(b, c, 4, 4)
        x_dense = F.interpolate(x_dense, size=(h, w), mode='bilinear', align_corners=False)
        
        x_flat = x.flatten(2).transpose(1, 2)
        energy = torch.mean(x_flat, dim=2)
        
        k = int(h * w * self.topk_ratio)
        topk_val, topk_idx = torch.topk(energy, k, dim=1) 
        
        batch_idx = torch.arange(b, device=x.device).unsqueeze(1).expand(-1, k)
        x_sparse_feat = x_flat[batch_idx, topk_idx]
        
        x_sparse_feat = self.sparse_mlp(x_sparse_feat)
        
        x_sparse = torch.zeros_like(x_flat)
        x_sparse[batch_idx, topk_idx] = x_sparse_feat
        x_sparse = x_sparse.transpose(1, 2).reshape(b, c, h, w)
        
        x_out = self.fusion(torch.cat([x_dense, x_sparse], dim=1))
        
        return x * self.sigmoid(x_out)

# =========================================================================
# 5. FSD-APAM Assembly
# =========================================================================
class FSD_APAM(nn.Module):
    def __init__(self, c1, reduction=4):
        super().__init__()
        c = c1
        
        self.local_conv3 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )
        
        self.fdd = FDD(c, kernel_size=5, wt_levels=1, wt_type="db1")
        self.ssd = SSD(k=3)
        
        self.local_fuse = nn.Sequential(
            nn.Conv2d(3 * c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

        self.acpa = ACPA(c)

        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(c // reduction, 8)
        self.gate_fc1 = nn.Linear(2 * c, mid)
        self.gate_fc2 = nn.Linear(mid, 2 * c)
        self.gate_act = nn.SiLU(inplace=True)

        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        local3 = self.local_conv3(x)
        local_wt = self.fdd(x)
        contrast = self.ssd(x)
        
        x_local = self.local_fuse(torch.cat([local3, local_wt, contrast], dim=1))

        x_global = self.acpa(x_local)

        gap_local = self.gap(x_local).view(b, c)
        gap_global = self.gap(x_global).view(b, c)

        gate_feat = torch.cat([gap_local, gap_global], dim=1)
        gate_mid = self.gate_act(self.gate_fc1(gate_feat))
        gate_vec = self.gate_fc2(gate_mid)

        gate_weights = F.softmax(gate_vec.view(b, c, 2), dim=2)
        w_local = gate_weights[:, :, 0].view(b, c, 1, 1)
        w_global = gate_weights[:, :, 1].view(b, c, 1, 1)

        x_fused = w_local * x_local + w_global * x_global

        return self.out_conv(x_fused)

if __name__ == "__main__":
    x = torch.randn(2, 64, 64, 64) 
    model = FSD_APAM(c1=64)
    out = model(x)
    print(f"Model Initialized.")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")