import torch
import torch.nn as nn
import copy


class GroupManager:
    """
    管理 Group 到 Subtask (Token) 的映射关系，实现分层控制逻辑。
    对应要点: 3. 分层控制粒度 (Hierarchical Control Granularity)
    """

    def __init__(self, first_head_sub_counts, device):
        """
        Args:
            first_head_sub_counts: List[int], 每个 Group 包含的 Subtask 数量。
                          例如 [3, 2] 表示 Group0 有3个子任务，Group1 有2个。
            device: torch.device
        """
        self.sub_counts = first_head_sub_counts.sub_counts
        self.num_groups = len(self.sub_counts)
        self.total_subtasks = sum(self.sub_counts)
        self.device = device

        # 生成 Group ID 到 Token ID 的映射索引，用于广播
        # group_ids: [0, 0, 0, 1, 1] 对应 5 个 tokens
        self.group_ids_per_token = []
        for g_idx, count in enumerate(first_head_sub_counts):
            self.group_ids_per_token.extend([g_idx] * count)
        self.group_ids_per_token = torch.tensor(self.group_ids_per_token, device=device)

    def broadcast_correction(self, delta_q_groups):
        """
        将 Group 维度的修正量广播到 Token 维度。
        Args:
            delta_q_groups: [Batch, Num_Groups, Hidden_Dim]
        Returns:
            delta_q_tokens: [Batch, Num_Subtasks, Hidden_Dim]
        """
        # 使用 index_select 或 gather 进行广播
        # [Batch, Num_Groups, H] -> [Batch, Num_Subtasks, H]
        B, G, H = delta_q_groups.shape
        # 扩展索引以匹配 batch 维度虽然 index_select 不需要，但为了通用性演示逻辑：
        # 这里直接在维度 1 (Group dim) 根据索引重复
        return delta_q_groups.index_select(1, self.group_ids_per_token)

    def compute_group_active_mask(self, subtask_active_mask):
        """
        逻辑 OR 机制：只要 Group 内有任意 Subtask 活跃，该 Group 视为活跃。
        Args:
            subtask_active_mask: [Batch, Num_Subtasks] (Bool or Float 0/1)
        Returns:
            group_active_mask_broadcasted: [Batch, Num_Subtasks, 1]
            注意：这里直接返回广播回 Token 维度的 Mask，方便后续计算
        """
        B, S = subtask_active_mask.shape

        # 1. 聚合：计算每个 Group 的活跃状态
        # 由于 PyTorch scatter_reduce 较新，这里用简化的循环或矩阵乘法演示原理
        # 构建聚合矩阵 M: [Num_Groups, Num_Subtasks]
        group_flags = torch.zeros(B, self.num_groups, device=self.device)

        start_idx = 0
        for g_idx, count in enumerate(self.sub_counts):
            # 获取该组对应的 subtasks mask片段
            sub_slice = subtask_active_mask[:, start_idx: start_idx + count]
            # OR 逻辑：max(mask) >= 1 (假设mask是0/1) 或者 any()
            group_is_active = (sub_slice > 0.5).any(dim=1).float()
            group_flags[:, g_idx] = group_is_active
            start_idx += count

        # 2. 广播：将 Group 的状态映射回 Token 维度
        # group_flags: [B, G] -> [B, S]
        group_active_mask_tokens = group_flags.index_select(1, self.group_ids_per_token)

        # 返回形状 [B, S, 1] 以便直接与 Hidden States 相乘
        return group_active_mask_tokens.unsqueeze(-1)


class RLEnv(nn.Module):
    def __init__(self, pretrained_model, config):
        """
        Args:
            pretrained_model: 预训练好的 Transformer 主干 (Query-centric ViT/BERT etc.)
            config: 配置对象
        """
        super().__init__()
        self.model = pretrained_model
        self.cfg = config
        self.device = next(pretrained_model.parameters()).device
        # 注意：如果 start_classify > 1，class_heads 依然是 0-indexed
        first_head_sub_counts = self.model.class_heads[0].sub_counts  # list 各个group中子任务的数量 [8, 8,...]

        # 初始化 Group 管理器
        self.group_manager = GroupManager(first_head_sub_counts, self.device)

        # 关键配置
        self.rl_start_layer = config.rl_start_layer  # RL 开始介入的层索引 (e.g., 6)
        self.total_layers = len(pretrained_model.blocks)  # Transformer 总层数
        self.freeze_classifier_flag = config.freeze_classifier  # 是否冻结分类头

        # 如果需要联合训练，可根据配置解冻分类头
        # 对应要点: 6. 自适应联合优化 (Adaptive Co-optimization)
        if self.freeze_classifier_flag:
            for param in self.model.classifier_head.parameters():
                param.requires_grad = False
        else:
            for param in self.model.classifier_head.parameters():
                param.requires_grad = True

        # 运行时状态缓存
        self.current_hidden_states = None
        self.current_layer_idx = 0
        self.subtask_active_mask = None  # [Batch, Num_Subtasks]
        self.final_logits = None  # [Batch, Num_Subtasks, Num_Classes_Per_Task]
        self.batch_indices = None  # 辅助索引

    def reset(self, batch_inputs):
        """
        对应要点: 5. 严格的阶段对齐 (Strict Stage Alignment)
        - 自动执行 Pre-rollout (Start layer 之前的所有层)
        - 初始化 Masks 和 Logits
        """
        # 1. 基础特征提取 (Stem / Embedding)
        # x: [Batch, Seq_Len, Dim] (包含 Query Tokens 和 Image/Context Tokens)
        x = self.model.patch_embed(batch_inputs)
        x = self.model.pos_drop(x)

        # 2. Pre-rollout: 执行 RL 介入前的 Transformer Block
        for i in range(self.rl_start_layer):
            x = self.model.blocks[i](x)

        # 3. 初始化状态
        self.current_hidden_states = x
        self.current_layer_idx = self.rl_start_layer  # RL 的 step 0 对应 start_layer

        B = x.shape[0]
        S = self.group_manager.total_subtasks

        # 初始化 Mask: 全部为 Active (1.0)
        self.subtask_active_mask = torch.ones(B, S, device=self.device)

        # 初始化最终结果容器 (用 0 填充或初始预测填充)
        # 假设分类头输出维度匹配
        dummy_logits = self.model.classifier_head(self.get_query_tokens(x))
        self.final_logits = torch.zeros_like(dummy_logits)

        self.batch_indices = torch.arange(B, device=self.device)

        # 返回初始观察值 (通常是当前的 Query Tokens 特征)
        return self.get_observation()

    def get_query_tokens(self, full_hidden_states):
        """
        从完整的序列中提取 Query Tokens。
        假设 Query Tokens 在序列的前 N 个位置。
        """
        num_q = self.group_manager.total_subtasks
        return full_hidden_states[:, :num_q, :]

    def step(self, action):
        """
        核心步进函数
        Args:
            action: Dict
                - 'stop': [Batch, Num_Subtasks] (Discrete: 0=Continue, 1=Stop)
                - 'correction': [Batch, Num_Groups, Hidden_Dim] (Continuous)
        """
        B, S = self.subtask_active_mask.shape

        # =============================================================
        # 1. 决策即时生效 (Immediate Decision Enforcement) - Action First
        # =============================================================

        # 解析动作
        stop_action = action['stop']  # [B, S]
        delta_q_groups = action['correction']  # [B, G, H]

        # 更新 Active Mask
        # 逻辑：如果当前步模型决定 Stop (1)，则 Mask 变为 0。
        # 如果之前已经是 0，保持 0 (Mask 只能单调递减)。
        # stop_action 为 1 表示停止，所以 active = (old_active) & (not stop)
        should_continue = (stop_action == 0).float()
        self.subtask_active_mask = self.subtask_active_mask * should_continue

        # 获取用于计算的 Boolean Mask
        active_bool_mask = (self.subtask_active_mask > 0.5)  # [B, S]

        # =============================================================
        # 2. 分层控制粒度 (Hierarchical Control)
        # =============================================================

        # A. 广播 Correction: Group -> Token
        delta_q_tokens = self.group_manager.broadcast_correction(delta_q_groups)  # [B, S, H]

        # B. 计算 Group 激活状态 (OR Logic)
        # [B, S, 1] - 如果 Token 所在的 Group 活跃，该值为 1
        group_active_mask = self.group_manager.compute_group_active_mask(self.subtask_active_mask)

        # =============================================================
        # 3. 应用修正 (Apply Correction)
        # =============================================================

        # 提取当前的 Query Tokens (引用，修改会影响 hidden_states)
        # 注意：在 PyTorch 中切片通常是 View，但为了安全建议显式写回
        all_tokens = self.current_hidden_states
        query_tokens = all_tokens[:, :S, :]  # [B, S, H]

        # 施加修正：
        # - 仅当 Group 处于 Active 状态时，才施加 Delta Q
        # - 这样保证了即使某个子任务 Stop 了，只要它的 Group 还在运作，它就能作为上下文被修正，
        #   从而服务于组内其他还在运行的子任务。
        effective_delta = delta_q_tokens * group_active_mask

        # 原地更新 Query Tokens
        query_tokens = query_tokens + effective_delta
        # 写回完整序列 (如果 query_tokens 不是 view)
        self.current_hidden_states[:, :S, :] = query_tokens

        # =============================================================
        # 4. 上下文完整性演化 (Context-Preserving Dynamics)
        # =============================================================

        # 全量更新：无论是否 Stop，所有 Token 进入 Transformer Layer 进行 Self-Attention
        # 这样 Deep Layers 能看到全局信息，避免 Context 缺失。
        layer_block = self.model.blocks[self.current_layer_idx]

        # Forward pass (Standard Transformer Block)
        # 输入形状: [B, Seq_Len, H], 输出: [B, Seq_Len, H]
        next_hidden_states = layer_block(self.current_hidden_states)

        # 更新状态
        self.current_hidden_states = next_hidden_states
        self.current_layer_idx += 1

        # =============================================================
        # 5. 结果灵活锁定 (Result Locking) & Early Exit
        # =============================================================

        # 计算当前层的 Logits
        # [B, S, H]
        current_query_features = self.get_query_tokens(self.current_hidden_states)
        # [B, S, Num_Classes]
        current_step_logits = self.model.classifier_head(current_query_features)

        # 锁定逻辑：
        # - 如果 Mask 为 1 (Active): 更新为 current_step_logits
        # - 如果 Mask 为 0 (Stopped): 保持 self.final_logits 不变 (即锁定在 Stop 那一刻的值)
        # 使用 torch.where 消除分支并支持梯度
        mask_expanded = self.subtask_active_mask.unsqueeze(-1)  # [B, S, 1]
        self.final_logits = torch.where(
            mask_expanded > 0.5,
            current_step_logits,
            self.final_logits
        )

        # =============================================================
        # 6. 步进结束处理
        # =============================================================

        # 检查是否所有任务都停止 或 达到最大层数
        all_stopped = (self.subtask_active_mask.sum() == 0)
        max_layers_reached = (self.current_layer_idx >= self.total_layers)

        done = all_stopped or max_layers_reached

        # 构造 Observation
        next_obs = self.get_observation()

        # Reward 计算 (示意，具体逻辑在 PPO Trainer 中)
        # 通常包含：准确率奖励 - 计算消耗惩罚
        reward = self.compute_reward(stop_action, done)

        info = {
            'active_mask': self.subtask_active_mask,
            'layer_idx': self.current_layer_idx,
            'final_logits': self.final_logits
        }

        return next_obs, reward, done, info

    def get_observation(self):
        """
        构造 RL Agent 的观察空间。
        通常包括：
        1. 当前 Query Tokens 的特征 (Capture current semantic)
        2. 当前的 Layer Index (Positional embedding for depth)
        3. 当前的 Active Mask (State info)
        """
        # [B, S, H]
        q_features = self.get_query_tokens(self.current_hidden_states).detach()
        # 这里使用 detach() 因为 RL 的输入通常不需要梯度回传到 Backbone (除非是 End-to-End RL)
        return {
            'features': q_features,
            'mask': self.subtask_active_mask,
            'layer_idx': self.current_layer_idx
        }

    def compute_reward(self, stop_action, done):
        """
        占位符：具体的 Reward 计算逻辑。
        应当基于 self.final_logits 与 Ground Truth 的 Loss 变化，以及 step 惩罚。
        """
        return 0.0
