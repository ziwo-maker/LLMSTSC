import os
from typing import Any, Optional

import torch
from PIL import Image

from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.TimeSeries_To_Image import time_series_to_simple_image


class TimeSeriesImageAdapter:
    """
    Convert time series inputs into image tensors with the same preprocessing
    logic as Model.vision_augmented_learner.
    中文：将时间序列输入转换为图像张量，复用与 Model.vision_augmented_learner 相同的预处理逻辑。
    """
    def __init__(
        self,
        image_size: int = 56,
        seq_len: Optional[int] = 25,
        periodicity: int = 25,
        norm_const: float = 0.4,
        learnable_image: bool = True,
        three_channel_image: bool = True,
        save_images: bool = False,
        device: Optional[torch.device] = None,
        input_dim: int = 3,
        hidden_dim: int = 48,
    ) -> None:
        """中文：初始化图像转换的配置与（可选）可学习图像模块。"""
        self.image_size = int(image_size)
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.periodicity = int(periodicity)
        self.norm_const = float(norm_const)
        self.learnable_image = bool(learnable_image)
        self.three_channel_image = bool(three_channel_image)
        self.save_images = bool(save_images)
        self.device = device
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learnable_image_module = None

        if self.learnable_image:
            out_channels = 3 if self.three_channel_image else 1
            self.learnable_image_module = LearnableTimeSeriesToImage(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_channels=out_channels,
                image_size=self.image_size,
                periodicity=self.periodicity,
            )
            if self.device is not None:
                self.learnable_image_module = self.learnable_image_module.to(self.device)

    def __call__(self, data: Any) -> torch.Tensor:
        """中文：使实例可直接被调用，等价于调用 generate。"""
        return self.generate(data)

    def _ensure_tensor(self, data: Any) -> torch.Tensor:
        """中文：将输入转换为形状为 (B, L, C) 的 float32 张量并对齐设备。"""
        if torch.is_tensor(data):
            x_enc = data
        else:
            x_enc = torch.as_tensor(data, dtype=torch.float32)

        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(0)
        if x_enc.ndim != 3:
            raise ValueError("Expected input shape (L, C) or (B, L, C).")

        if x_enc.dtype != torch.float32:
            x_enc = x_enc.float()

        if self.device is not None and x_enc.device != self.device:
            x_enc = x_enc.to(self.device)

        return x_enc

    def _align_device(self, x_enc: torch.Tensor) -> torch.Tensor:
        """中文：若使用可学习模块，将输入移动到该模块所在设备。"""
        if self.learnable_image_module is None:
            return x_enc
        module_device = next(self.learnable_image_module.parameters()).device
        if x_enc.device != module_device:
            x_enc = x_enc.to(module_device)
        return x_enc

    def _context_len(self, x_enc: torch.Tensor) -> int:
        """中文：根据配置与输入长度确定上下文长度。"""
        if self.seq_len is None or self.seq_len <= 0:
            return x_enc.shape[1]
        if x_enc.shape[1] != self.seq_len:
            return x_enc.shape[1]
        return self.seq_len

    def _normalize_input(self, x_enc: torch.Tensor):
        """中文：按序列维度做去均值与标准化，返回归一化张量及均值/方差。"""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.norm_const
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    @staticmethod
    def _normalize_images(images: torch.Tensor) -> torch.Tensor:
        """中文：把图像张量线性缩放到 [0, 255] 并转为 uint8。"""
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        images = (images - min_vals) / scale
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

    def vision_augmented_learner(
        self,
        x_enc: torch.Tensor,
        image_size: int,
        context_len: int,
        periodicity: int,
    ) -> torch.Tensor:
        """中文：将时序特征映射为图像并归一化，必要时保存到磁盘。"""
        if self.learnable_image:
            if self.learnable_image_module is None:
                out_channels = 3 if self.three_channel_image else 1
                self.learnable_image_module = LearnableTimeSeriesToImage(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_channels=out_channels,
                    image_size=image_size,
                    periodicity=periodicity,
                )
            images = self.learnable_image_module(x_enc)
        else:
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)

        images = self._normalize_images(images)

        if self.save_images:
            self.save_images_to_disk(images)

        return images

    def generate(self, data: Any) -> torch.Tensor:
        """中文：执行完整流程（含输入归一化）并生成图像张量。"""
        x_enc = self._ensure_tensor(data)
        x_enc = self._align_device(x_enc)
        x_enc, _, _ = self._normalize_input(x_enc)
        context_len = self._context_len(x_enc)
        return self.vision_augmented_learner(x_enc, self.image_size, context_len, self.periodicity)

    def generate_from_normalized(self, data: Any) -> torch.Tensor:
        """中文：假设输入已归一化，直接生成图像张量。"""
        x_enc = self._ensure_tensor(data)
        x_enc = self._align_device(x_enc)
        context_len = self._context_len(x_enc)
        return self.vision_augmented_learner(x_enc, self.image_size, context_len, self.periodicity)

    def generate_images(self, data: Any, normalized: bool = False, return_tensor: bool = False):
        """中文：生成图像（可选已归一化输入）并选择返回张量或 PIL 图片。"""
        if normalized:
            images = self.generate_from_normalized(data)
        else:
            images = self.generate(data)
        if return_tensor:
            return images
        return self.to_pil(images)

    def to_pil(self, images: torch.Tensor):
        """中文：将图像张量批量转换为 PIL.Image 列表。"""
        if not torch.is_tensor(images):
            return None
        images = images.detach().cpu()
        pil_images = []
        for img in images:
            if img.shape[0] == 1:
                array = img.squeeze(0).numpy()
                mode = "L"
            else:
                array = img.permute(1, 2, 0).numpy()
                mode = "RGB"
            pil_images.append(Image.fromarray(array, mode=mode))
        return pil_images

    def save_images_to_disk(self, images: torch.Tensor) -> None:
        """中文：将图像张量批量保存为 PNG 到固定目录。"""
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)

        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.detach().cpu()
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
                mode = "RGB"
            elif img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
                mode = "L"
            else:
                continue

            try:
                img = Image.fromarray(img_tensor.numpy(), mode=mode)
                img.save(os.path.join(save_dir, f"image_{i}.png"))
            except Exception:
                continue
