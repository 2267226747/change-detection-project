# 定义 Actor (Policy) 和 Critic (Value) 网络结构
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    状态编码器：负责将多模态状态融合为特征向量。
    处理单路数据 (例如 q1 + v1 + logits + time)，q2 + v2 复用此实例。
    """

    def __init__(self, cfg):
        super().__init__()
        # --- 1. 读取配置 ---
        self.model_dim = getattr(cfg.model, 'dim', 1024)  # Query 的维度
        self.vis_dim = getattr(cfg.model, 'vision_dim', 768)  # VisionEncoder 输出的维度
        # 注意：这里的 total_subtasks 需要在初始化前确定 (例如 31)
        # 如果 cfg 中没有，可以在实例化时通过 kwargs 传入，或者在 cfg 中预先定义
        self.num_classes = getattr(cfg.model, 'total_subtasks', 31)
        self.max_time_steps = getattr(cfg, 'num_layers', 24) + 1

        # 融合策略: 'concat' (默认) 或 'attention'
        self.fusion_type = getattr(cfg.rl, 'fusion_type', 'concat')

        # --- 2. 特征投影层 (Align Dimensions) ---
        # 将视觉特征映射到 Transformer 维度
        self.vis_proj = nn.Linear(self.vis_dim, self.model_dim)

        # 将 Logits 映射到 Transformer 维度
        self.logit_proj = nn.Linear(self.num_classes, self.model_dim)

        # 时间步 Embedding
        self.time_embed = nn.Embedding(self.max_time_steps, self.model_dim)

        # --- 3. 融合层 ---
        if self.fusion_type == 'attention':
            # 使用 Cross Attention 融合
            # Query = Query Token, Key/Value = [Visual, Logits, Time]
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=4,
                batch_first=True
            )
            self.norm = nn.LayerNorm(self.model_dim)
        else:
            # 使用 Concat + MLP 融合
            # 输入 = Query(D) + Vis(D) + Logits(D) + Time(D) = 4D
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.model_dim * 4, self.model_dim * 2),
                nn.LayerNorm(self.model_dim * 2),
                nn.GELU(),
                nn.Linear(self.model_dim * 2, self.model_dim),
                nn.LayerNorm(self.model_dim)
            )

    def forward(self, query_tokens, visual_context, logits, time_step):
        """
        Args:
            query_tokens: [Batch, N, Dim] - 当前的 Query 序列
            visual_context: [Batch, Vis_Dim] - 全局视觉特征 (Global Avg Pooled)
            logits: [Batch, Total_Subtasks] - 展平后的分类置信度
            time_step: [Batch] - 当前步数索引
        Returns:
            fused_feat: [Batch, N, Dim]
        """
        B, N, D = query_tokens.shape

        # 1. 投影与扩展 (Projection & Expansion)

        # Visual: [B, V_Dim] -> [B, D] -> [B, 1, D] -> [B, N, D]
        vis_feat = self.vis_proj(visual_context).unsqueeze(1).expand(-1, N, -1)

        # Logits: [B, C] -> [B, D] -> [B, 1, D] -> [B, N, D]
        logit_feat = self.logit_proj(logits).unsqueeze(1).expand(-1, N, -1)

        # Time: [B] -> [B, D] -> [B, 1, D] -> [B, N, D]
        time_feat = self.time_embed(time_step).unsqueeze(1).expand(-1, N, -1)

        # 2. 融合 (Fusion)
        if self.fusion_type == 'attention':
            # 构造 KV 序列: Stack [Visual, Logits, Time] -> [B, 3, D]
            # 注意：这里我们让每个 Query Token 去 attend 全局的 Vis/Logit/Time 信息
            kv_seq = torch.stack([
                self.vis_proj(visual_context),
                self.logit_proj(logits),
                self.time_embed(time_step)
            ], dim=1)  # [B, 3, D]

            # Cross Attention: Q=query_tokens, K=V=kv_seq
            attn_out, _ = self.fusion_layer(query=query_tokens, key=kv_seq, value=kv_seq)

            # Residual Connection
            fused_feat = self.norm(query_tokens + attn_out)

        else:  # 'concat'
            # 简单拼接: [B, N, 4*D]
            concat_feat = torch.cat([query_tokens, vis_feat, logit_feat, time_feat], dim=-1)
            fused_feat = self.fusion_layer(concat_feat)

        return fused_feat


class ActorNetwork(nn.Module):
    """
    策略网络：接收状态，输出 Delta Query。
    """

    def __init__(self, cfg):
        super().__init__()
        self.model_dim = getattr(cfg.model, 'dim', 1024)

        # 1. 状态编码器
        self.encoder = StateEncoder(cfg)

        # 2. 动作生成头 (Action Head)
        # 输入融合后的特征，输出 Delta
        self.head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
            nn.Tanh()  # 限制输出范围在 [-1, 1]
        )

        # 3. 可学习缩放因子 (初始化为0)
        # 这样在训练初期，delta 接近 0，网络行为等同于原始预训练模型
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, query, visual_context, logits, time_step):
        """
        前向传播
        Args:
            query: [B, N, D]
            visual_context: [B, Vis_Dim]
            logits: [B, Total_Subtasks]
            time_step: [B]
        Returns:
            delta_query: [B, N, D]
        """
        # 1. 编码
        fused_state = self.encoder(query, visual_context, logits, time_step)

        # 2. 生成动作
        raw_delta = self.head(fused_state)

        # 3. 缩放
        return raw_delta * self.scale
