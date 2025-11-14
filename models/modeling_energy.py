import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_calm import CALMConfig
from .modeling_calm import CALM, CustomCausalLMOutput
from .configuration_autoencoder import AutoencoderConfig
from .modeling_autoencoder import Autoencoder
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel,LlamaModel,LlamaRMSNorm
import random

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

class MLPBlock(nn.Module):
    """
    A single residual block for the MLP-based generative head.

    This block refines an input representation 'x' (e.g., a noise vector embedding)
    by conditioning it on a context vector 'y' (e.g., the Transformer's hidden state).
    It uses a gated MLP structure for effective feature fusion and transformation.

    单一残差块，负责把噪声向量 ε_l  迭代 refine 成 ε_{l+1}，
    条件向量 y（Transformer hidden state）通过门控 MLP 注入。
    对应论文 §3.3.3 中描述的：
        ε_{l+1} = ε_l + MLP([ε_l ; y])
    """
    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(       
            # 门控 MLP：2*channels → SiLU → channels → SiLU → 2*channels
            nn.Linear(2 * channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True)
        )
        self.gate_act = nn.SiLU()
        self.down_proj = nn.Linear(channels, channels, bias=True)


    def forward(self, x, y):
        # [ε_l ; y] 过MLP
        h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))     # [B, 2* channels]
        # SwiGLU门控机制：切成 gate 与 up
        gate_proj, up_proj = torch.chunk(h, 2, dim = -1)    # [2 * channels] -> [channels], [channels]
        gate_proj = self.gate_act(gate_proj)
        step = self.down_proj(gate_proj * up_proj)

        return x + step   # 残差链接


# =============================================================================
# 2.  FinalLayer – 把最后的 ε_L 投影到目标 latent 维度
# =============================================================================
class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(model_channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(model_channels, model_channels, bias=True),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels, bias=True)
        )

    def forward(self, x):
        h = self.linears(self.in_ln(x))
        return h

# =============================================================================
# 3.  MLPGenerator – 对应论文图 2 右侧的 Energy-Based Generative Head
# =============================================================================
class MLPGenerator(nn.Module):
    """
    MLP-based generative head. 
    This module takes a Transformer hidden state and a random noise vector as input
    and generates a continuous latent vector prediction. 
    It consists of a stack of MLPBlocks that iteratively refine the noise conditioned on the hidden state.  

    生成头：给定 Transformer 隐状态 y，先生成随机噪声 ε_0～U(-0.5,0.5)，
    经过 L 个 MLPBlock 迭代 refine，最终投影为连续向量 z_i。
    对应 §3.3.3 的 ε_0 → ε_L → z_i 过程。
    """
    def __init__(self, config):
        super().__init__()
        self.noise_size = config.noise_size     # d_noise  论文中 |ε|
        self.noise_embd = nn.Linear(config.noise_size, config.hidden_size)  # [B, noise_size] -> [B, hidden_size]
        self.hidden_embd = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_hidden = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm_noise = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 堆叠 num_mlp_layers 个 MLPBlock（论文 L = 1/4 * transformer layers）
        mlp_blocks = []
        for i in range(config.num_mlp_layers):
            mlp_blocks.append(MLPBlock(
                config.hidden_size,
            ))
        self.mlp_blocks = nn.ModuleList(mlp_blocks)
        # 最终投影到 latent 维度 l（对应论文符号 l）
        self.final_layer = FinalLayer(config.hidden_size, config.latent_size)

    def initialize_weights(self):
        """
        将最后一层初始化为 0，保证初始时生成向量接近 0，训练稳定。
        """
        nn.init.constant_(self.final_layer.linears[-1].weight, 0)
        nn.init.constant_(self.final_layer.linears[-1].bias, 0)
        return
    
    def sample(self, hidden_states):
        """
        hidden_states : y_i-1  (batch, hidden)  Transformer 输出
        returns       : z_i    (batch, latent)  连续向量
        """
        # 1. 采样噪声 ε_0 ～ U(-0.5,0.5)
        # Prepare noise for sampling
        noise = torch.rand((*hidden_states.shape[:-1], self.noise_size),
                   dtype=hidden_states.dtype, device=hidden_states.device) - 0.5

        # 2. 嵌入 + LayerNorm
        noise_embds = self.norm_noise(self.noise_embd(noise))
        hidden_states = self.norm_hidden(self.hidden_embd(hidden_states))

        # 3. 迭代 refine
        for block in self.mlp_blocks:
            noise_embds = block(noise_embds, hidden_states)

        # 4. 投影到 latent 空间
        latent_prediction = self.final_layer(noise_embds)
        return latent_prediction

class EnergyTransformer(CALM):
    """
    The main Continuous Autoregressive Language Model (CALM).
    This model integrates a standard Transformer backbone with a continuous generative head.
    It operates by predicting continuous vectors, each representing a chunk of K tokens.
    This model is trained with a likelihood-free Energy Score objective.

    训练目标：最大化 Energy Score（等价于最小化负 Energy Score）。
    评估：使用 BrierLM（无需似然）。
    """
    config_class = CALMConfig 

    def __init__(self, config):
        super().__init__(config)

         # 加载预训练自编码器（冻结）
        self.ae_config = AutoencoderConfig.from_pretrained(config.ae_path)
        self.ae_model = Autoencoder.from_pretrained(
            config.ae_path,
            config=self.ae_config,
        )
        # Freeze the autoencoder weights during CALM training
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()
        
        # Transformer 主干
        self.transformer = LlamaModel(config)
        # 生成头
        self.mlp_generator = MLPGenerator(config)
        self.generative_head = self.mlp_generator

        # 特殊 token
        self.padding_idx = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.patch_size = config.patch_size

        # 输入压缩模块：把 K 个 token 嵌入拼在一起后映射到 hidden 【input compression mlp】
        # Input compression module: maps K token embeddings to a single vector
        self.embed_proj = nn.Sequential(
            nn.Linear(self.patch_size * config.hidden_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6)
        )
        # Initialize weights and apply final processing
        self.post_init()
        self.mlp_generator.initialize_weights()
        self.noise_size = config.noise_size
        self.beta = config.beta
        self.num_samples = config.num_samples
    

    # -------------------------------------------------------------------------
    # 工具函数：计算 ||x-y||^β
    # -------------------------------------------------------------------------
    def distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.beta)
    
    # -------------------------------------------------------------------------
    # Energy Score 实现（对应论文公式 9）
    # -------------------------------------------------------------------------
    def energy_score(self, x, mean, log_std):
        """
        x     : (n_x, batch, latent)  模型生成的 N 个样本  z̃_i
        mean  : (batch, latent)       自编码器 posterior 的 μ
        log_std:(batch, latent)       自编码器 posterior 的 logσ
        返回   : (batch,)             Energy Score 值（越大越好）
        """
        n_x = x.shape[0]
        #  pairwise ||z̃_i - z̃_j||^β
        x_i = x.unsqueeze(1)  # (n_x, 1, batch, ...)
        x_j = x.unsqueeze(0)  # (1, n_x, batch, ...)
        distance_matrix = self.distance(x_i, x_j)
        # 第一项：E[||z̃_i - z̃_j||^β]
        distance_x = distance_matrix.sum(dim=(0, 1)) / (n_x * (n_x - 1))

        # 从 posterior 采样 M=100 个目标向量 y~q(z|x)
        std = torch.exp(log_std)
        n_y = 100
        eps = torch.randn((n_y, *mean.shape), device=mean.device)
        y = mean + eps * std  # (n_y, batch, latent)
        y = y

        # 第二项：E[||z̃_i - y||^β]
        x_ = x.reshape(n_x, 1, *x.shape[1:])  # (n_x, 1, batch, latent)
        y_ = y.reshape(1, n_y, *y.shape[1:])  # (1, n_y, batch, latent)
        distance_y = self.distance(x_, y_).mean(dim=(0, 1))

        # Energy Score = ① - 2*②
        score = distance_x - distance_y * 2
        return score

    # -------------------------------------------------------------------------
    # forward – 训练 & 评估入口
    # -------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        input_ids: (batch, seq)  已补全成 K 的倍数
        labels:   (batch, seq)  用于重建目标，与 input_ids 对齐
        返回      CustomCausalLMOutput，包含 loss 与（评估时）brier 指标
        """

        batch_size, seq_length = input_ids.size()
        patch_size = self.patch_size
        latent_length = seq_length // patch_size

        # 只对「下一个 patch」做预测，因此 labels 去掉第一个 patch
        labels = labels[:, patch_size:]
        mask = labels.ne(-100)          # 计算 loss 时忽略 pad
        labels = labels[mask].unsqueeze(0)

        # 1. 用冻结的自编码器得到目标 posterior  q(z|x_{1:K})
        #    latent_states = [μ, logσ]  对应论文公式
        latent_states = self.ae_model.encoder(input_ids=labels)
        latent_states = latent_states.squeeze(0)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)

        # 2. Transformer 输入：把前一个 patch 的 K 个 token 嵌入压缩成单向量
        inputs_embeds = self.transformer.embed_tokens(input_ids)\
                                        .reshape(batch_size, latent_length, -1)[:, :-1, :]
        inputs_embeds = self.embed_proj(inputs_embeds)

        # 3. 得到 Transformer 隐状态 y_i-1
        outputs = self.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]                      # (batch, L-1, hidden)
        # 只保留有效 patch 位置
        patch_mask = mask.reshape(batch_size, latent_length - 1, patch_size)[:, :, 0]
        hidden_states = hidden_states[patch_mask]       # (total_patches, hidden)

        # 4. 生成 N 个样本 z̃_i
        hidden_states_repeated = hidden_states.unsqueeze(0).repeat(self.num_samples, 1, 1)
        latent_predictions = self.generative_head.sample(hidden_states_repeated)

        # 5. 计算负 Energy Score 作为 loss
        loss = -self.energy_score(latent_predictions, mean, log_std)
        loss = loss.mean()

        # 6. 评估阶段额外计算 BrierLM
        if not self.training:
            return self.eval_brier(latent_predictions,
                                   input_ids[:, patch_size:],  # 去掉第一个 patch 的 token
                                   outputs, loss)

        return CustomCausalLMOutput(loss=loss)