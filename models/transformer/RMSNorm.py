import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. 保存原始数据类型 (如 torch.bfloat16)
        input_dtype = x.dtype

        # 2. 【关键】强制转为 FP32 进行统计量计算
        # 防止 x**2 产生精度丢失或溢出
        x = x.float()

        # 3. 在 FP32 下计算 RMS
        var = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)

        # 4. 转回原始类型 (BF16)
        # 5. 乘上可学习的权重 (weight 会由 autocast 自动处理，或者此处广播乘法)
        return self.weight * x_normed.to(input_dtype)