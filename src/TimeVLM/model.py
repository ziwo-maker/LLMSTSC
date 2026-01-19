import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image

# Ensure project root is on sys.path when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.TimeVLM.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.models_mae import *
from transformers.models.vilt import *

# Patch 记忆库：维护历史 patch 的特征，用于近邻检索与记忆增强
class PatchMemoryBank:
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        """
        Initialize the patch memory bank.
        
        Args:
            max_size (int): Maximum number of patches to store.
            patch_size (int): Size of each patch.
            feature_dim (int): Dimensionality of each patch feature.
            device (torch.device): Device to store memory bank on (CPU/GPU).
        """
        # 保存基础配置：容量、patch 尺寸、特征维度、设备
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        # 以循环缓冲区方式维护 patch 表示（默认 [max_size, d_model]）
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)  # [100, d_model]
        # 当前写入位置指针
        self.ptr = 0

    def update(self, new_patches):
        """
        Update the patch memory bank with new patches using circular buffer strategy.
        
        Args:
            new_patches (Tensor): New patches to add to the memory bank.
        """
        n = new_patches.size(0)
        # 将 patch 序列压缩为单个向量表示（对 token 维做均值）
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]
        
        if self.ptr + n > self.max_size:
            # Wrap around if the memory bank is full
            remaining_space = self.max_size - self.ptr
            self.patches[self.ptr:] = new_patches_flat[:remaining_space]        
            remaining_patches = n - remaining_space
            if remaining_patches >= self.max_size:
                # 新数据超过容量：仅保留最后 max_size 个
                self.patches[:] = new_patches_flat[-self.max_size:]
                self.ptr = 0
            else:
                # 环绕写入剩余位置
                self.patches[:remaining_patches] = new_patches_flat[remaining_space:]
                self.ptr = remaining_patches
        else:
            # 正常顺序写入
            self.patches[self.ptr:self.ptr + n] = new_patches_flat
            self.ptr += n



class Model(nn.Module):
    """
    Time-VLM model with image and text modalities for enhanced time series forecasting.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        # VLM 管理器：负责视觉-语言模型初始化与推理
        self.vlm_manager = VLMManager(config)
        # Require GPU for this run; fail fast if unavailable.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This model requires GPU.")
        if not getattr(self.config, "use_gpu", False):
            raise RuntimeError("This model requires GPU. Please run with --use_gpu True.")
        self.device = torch.device("cuda:{}".format(self.config.gpu))
        # 是否启用记忆门控融合
        self.use_mem_gate = config.use_mem_gate
        
        # Initialize patch memory bank

        # 构建各子模块
        self._init_modules(config)
        # VLM 具体模型实例
        self.vlm_model = self.vlm_manager.model

    def _init_modules(self, config):
        # 将时间序列切分为 patch 并做嵌入
        self.patch_embedding = PatchEmbedding(
            config.d_model, 
            config.patch_len, 
            config.stride, 
            config.padding, 
            config.dropout
        )
        # 计算 flatten 后的特征维度：n_patches * d_model
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        # 将 [n_patches, d_model] 拉平成向量
        self.flatten = nn.Flatten(start_dim=-2)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # 多模态预测头：对融合后的 d_model 做 pred_len 映射
        self.multimodal_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Vision enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Cross-modal attention for feature enhancement
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Memory fusion gate
        if self.use_mem_gate:
            # 记忆门控：根据局部/全局记忆自适应加权
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
                nn.Softmax(dim=-1)
            )

        # Memory-related modules
        # 局部记忆 MLP：对检索到的 patch 进行非线性变换
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        # 全局记忆注意力：捕获 patch 间依赖关系
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 学习式图像生成模块（可将时间序列转为图像特征）
        #这个是转化为图像的重点
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        
        # 可学习的门控标量
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
        # LayerNorm 统一特征尺度
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward_prediction(self, vision_embeddings, n_vars):
        # 仅使用视觉嵌入进行预测
        multimodal_features = self.multimodal_enhancement(vision_embeddings)  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(multimodal_features)  # [B, n_vars, d_model]
        predictions = self.multimodal_head(multimodal_features)  # [B, n_vars, pred_len]
        return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        # 确保输入在指定设备上
        x_enc = x_enc.to(self.device)
        
        # Normalize input
        # 标准化时间序列，记录均值与方差用于反归一化
        x_enc, means, stdev = self._normalize_input(x_enc)
        
        # Convert time series data to images
        # 生成图像表示
        images = self.vision_augmented_learner(x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity)
        
        # Process images with the VLM
        # 仅提取视觉嵌入
        vision_embeddings = self.vlm_manager.process_images(images)
        
        # Main prediction branch
        # 进入主预测分支
        predictions = self.forward_prediction(vision_embeddings, D)
        
        # Denormalize output
        # 将预测结果反归一化回原始尺度
        y = self._denormalize_output(predictions, means, stdev)
        return y

    def _normalize_input(self, x):
        # 计算序列均值并中心化
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        # 计算标准差并缩放，避免数值过大
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        # 还原尺度：乘回标准差并加回均值
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        # 根据配置选择可学习图像模块或固定的时间序列转图像方法.
        #z
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize images to [0, 255] as uint8
        # 归一化到 [0, 255] 的 uint8，便于 VLM 处理
        images = self._normalize_images(images)
        
        # Optionally save images
        # 若开启保存选项，则将图像落盘
        if self.config.save_images:
            self.save_images(images)

        return images
    
    @staticmethod
    def _normalize_images(images):
        """
        Normalize image tensors to [0, 255] as uint8.
        Assumes images are in [0, 1] or need to be scaled.
        
        Args:
        - images (Tensor): Input images with shape [B, C, H, W]
        
        Returns:
        - Tensor: Normalized images as uint8 with shape [B, C, H, W]
        """
        # Compute min and max per image across all channels and spatial dimensions
        # 逐图像计算全局最小/最大值
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Avoid division by zero by adding a small epsilon
        # 防止除零
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        # 线性缩放至 [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        # 缩放到 [0, 255] 并转 uint8
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        
        return images

    @torch.no_grad()
    def save_images(self, images):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved with shape [B, C, H, W]
        """
        # 保存目录
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            # Move to CPU and convert to numpy
            # 转到 CPU，并转为 numpy 便于 PIL 保存
            img_tensor = img_tensor.cpu().numpy()
            
            # Check channel count and handle accordingly
            # 根据通道数选择 RGB 或灰度模式
            if img_tensor.shape[0] == 3:
                # RGB image: Convert from [C, H, W] to [H, W, C]
                img_tensor = np.transpose(img_tensor, (1, 2, 0))
                mode = 'RGB'
            elif img_tensor.shape[0] == 1:
                # Grayscale image: Convert from [C, H, W] to [H, W]
                img_tensor = np.squeeze(img_tensor, 0)
                mode = 'L'
            else:
                print(f"Warning: Unexpected number of channels {img_tensor.shape[0]} for image {i}. Skipping...")
                continue
            
            # Ensure data type is uint8
            # 确保数据类型为 uint8
            if img_tensor.dtype != np.uint8:
                img_tensor = img_tensor.astype(np.uint8)
            
            # Create PIL image and save
            # 保存为 PNG 文件
            try:
                img = Image.fromarray(img_tensor, mode=mode)
                img.save(os.path.join(save_dir, f"image_{i}.png"))
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue
