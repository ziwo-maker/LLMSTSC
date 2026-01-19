from sqlite3 import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import numpy as np

class TimeSeriesVisualizer:
    """时序到图像转换的可视化工具"""
    
    @staticmethod
    def plot_feature_maps(features, title="Feature Maps"):
        """绘制中间层特征图"""
        if not isinstance(features, torch.Tensor):
            return
            
        # 转为 numpy 并做 0-1 归一化，便于显示
        features = features.detach().cpu().numpy()
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # 只可视化第一个 batch 的样本
        num_channels = min(4, features.shape[1])
        if num_channels == 1:
            plt.figure(figsize=(5, 5))
            plt.imshow(features[0, 0], cmap='viridis')
            plt.axis('on')
            plt.title('Channel 1')
        else:
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(features[0, i], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {i+1}')
        plt.suptitle(title)
        plt.tight_layout()
        
        # 保存图片到本地
        import os
        os.makedirs('ts-images/ts-visualizer', exist_ok=True)
        filename = f"ts-images/ts-visualizer/{title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        plt.close()

    @staticmethod
    def plot_attention(attention_map, title="Attention Map"):
        """绘制注意力权重图"""
        if not isinstance(attention_map, torch.Tensor):
            return
            
        attention_map = attention_map.detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map[0, 0], cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()

class LearnableTimeSeriesToImage(nn.Module):
    """可学习的时序到图像转换模块"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        # 基本超参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D 卷积：在时间维上提取局部模式
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D 卷积：在 (变量维, 时间维) 上提取二维结构
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """将输入时序转为图像张量 [B, output_channels, H, W]"""
        # x_enc: [B, L, D]，B 批大小，L 序列长度，D 变量维度
        B, L, D = x_enc.shape
        
        # 生成周期性编码（sin/cos），用于注入时间位置信息
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        # 扩展到变量维，形状 [B, L, D, 2]
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)
        
        # FFT 频域幅值编码，强调周期/频率成分
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        # rfft 输出长度可能短于 L，补齐到 L 以便与时域对齐
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # 拼接原始值 + 频域幅值 + 周期编码
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, D, 4]

        # 调整维度以适配 1D 卷积：将变量维展平到 batch
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D 卷积：在 (D, L) 平面上提取结构特征
        x_enc = x_enc.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(self.conv2d_2(x_enc))
        
        # 双线性插值到目标图像尺寸
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # 输出形状: [B, output_channels, H, W]
        return x_enc
