import torch
import torch.nn as nn


class QueryGenerator(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: 全局配置对象 (cfg.model.query_token)
        """
        super().__init__()

        cfg = cfg.model.query_token
        self.num_tasks = getattr(cfg, 'task_nums', 4)
        self.tokens_per_task = getattr(cfg, 'tokens_per_task', 256)
        self.dim = getattr(cfg, 'dim', 1024)

        # [修改] 移除 self.batch_size 的硬编码读取
        # Batch Size 应该由输入数据的实际大小决定，而不是配置文件
        # self.batch_size = getattr(cfg, 'batch_size', 16)

        # --- 组件 1: 语义属性 Embedding (Group Attribute) ---
        # 形状: [4, D]
        # 作用: 决定大方向 (建筑/绿化...)
        # 策略: 正交初始化 (Orthogonal) - 确保不同组之间差异最大化
        # [注意] nn.Parameter 默认为 FP32，这是正确的，请勿修改
        self.group_embed = nn.Parameter(torch.empty(self.num_tasks, self.dim))
        nn.init.orthogonal_(self.group_embed)

        # --- 组件 2: 个体 Embedding (Individual Variance) ---
        # 形状: [4, 256, D]
        # 作用: 决定个体差异 (关注纹理? 关注边缘? 关注左上角?)
        # 策略: 随机初始化 (Random Normal) - 初始值要小，依附于 Group
        self.token_embed = nn.Parameter(torch.randn(self.num_tasks, self.tokens_per_task, self.dim) * 0.02)

    def forward(self, current_batch_size):
        """
        组合 Group 和 Token Embedding 生成最终 Query
        Args:
            current_batch_size (int): 当前 batch 的实际大小 (来自 input_images.shape[0])
        """
        # 1. 扩展 Group Embed: [4, D] -> [4, 1, D] -> [4, 256, D]
        # 利用 broadcasting 让组内共享同一个语义中心
        # [保留] 这里是在 FP32 下进行的广播，精度最高
        group_semantic = self.group_embed.unsqueeze(1).expand(-1, self.tokens_per_task, -1)

        # 2. 叠加个体差异
        # Q = Base + Delta
        # [4, 256, D]
        final_query = group_semantic + self.token_embed

        # 3. 展平为 transformer 需要的形状
        # [4, 256, D] -> [1024, D]
        final_query = final_query.reshape(-1, self.dim)

        # 4. 扩展 Batch 维度
        # [B, token_nums, D]
        # [修改] 使用传入的 current_batch_size，适应验证集末尾和单张推理
        # [修改] 移除 .to(torch.bfloat16)，由外部 autocast 自动控制计算精度
        return final_query.unsqueeze(0).repeat(current_batch_size, 1, 1)