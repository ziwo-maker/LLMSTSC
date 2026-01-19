import einops
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    # 解析输入张量的三维语义：批大小、序列长度、变量数量
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # 为了能够按周期重排成二维图像，先将序列长度补齐到 periodicity 的整数倍
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # 将时间序列维度放到最后一维，方便后续按周期切分
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # 左侧复制填充，避免引入无意义的零值
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # 按照 periodicity 切成二维块：频率维 f 与周期维 p
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # 双线性插值缩放到统一图像尺寸
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # 复制为 3 通道，形成伪 RGB 图像
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)  # [B * nvars, 3, H, W]

    # 恢复批与变量维度，并在变量维度上取平均以融合多变量信息
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images


def time_series_to_image_with_fft_and_wavelet(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors using FFT and Wavelet transforms.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    def _apply_fourier_transform(x_2d):
        """
        Apply Fourier transform to the input 2D time series data.
        """
        # 在最后一维进行 FFT，捕获周期维的频谱特征
        x_fft = torch.fft.fft(x_2d, dim=-1)
        # 取幅值作为频谱强度，丢弃相位信息
        x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
        return x_fft_abs

    def _apply_wavelet_transform(x_2d):
        """
        Apply wavelet transform to the input 2D time series data.
        """
        # Haar 小波分解：J=1 表示一层分解
        dwt = DWTForward(J=1, wave='haar')
        # cA: Low-frequency components, cD: High-frequency components
        cA, cD = dwt(x_2d)  # [B * nvars, 1, f, p]
        # 高频分量列表取第一层，并去掉多余的通道维
        cD_reshaped = cD[0].squeeze(1)  # [B * nvars, 3, f, p]
        # 拼接低频与高频特征，形成更丰富的局部变化表示
        wavelet_result = torch.cat([cA, cD_reshaped], dim=1)  # [B * nvars, 4, f, p]
        # 在通道维上取平均，压缩回单通道表示
        wavelet_result = wavelet_result.mean(dim=1, keepdim=True)  # [B * nvars, 1, f, p]
        return wavelet_result
    
    # 解析输入张量的三维语义：批大小、序列长度、变量数量
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # 为了能够按周期重排成二维图像，先将序列长度补齐到 periodicity 的整数倍
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # 将时间序列维度放到最后一维，方便后续按周期切分
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # 左侧复制填充，避免引入无意义的零值
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # 按照 periodicity 切成二维块：频率维 f 与周期维 p
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # 双线性插值缩放到统一图像尺寸，作为时间域通道输入
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # 生成频域与小波域特征
    x_fft = _apply_fourier_transform(x_2d)
    x_wavelet = _apply_wavelet_transform(x_2d)
    # 将频域与小波域特征插值到统一尺寸
    x_resized_fft = F.interpolate(x_fft, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x_resized_wavelet = F.interpolate(x_wavelet, size=(image_size, image_size), mode='bilinear', align_corners=False)
    # 拼接三个通道：时间域 + 频域 + 小波域
    images = torch.concat([x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1)  # [B * nvars, 3, H, W]

    # 恢复批与变量维度，并在变量维度上取平均以融合多变量信息
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images
