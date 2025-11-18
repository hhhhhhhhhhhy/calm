# coding=utf-
"""
Continuous Autoencoder 实现

  - Encoder: 将 K 个离散 token → 连续向量  z ∈ R^l       （论文 l=128 时 K=4）
  - Decoder: 将 z 还原为 K 个 token，重建精度 ≥99.9 %
  - 训练目标: 重构 CE  +  β·KL(q_φ(z|x)||N(0,I))
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from .configuration_autoencoder import AutoencoderConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel,LlamaRMSNorm, LlamaMLP

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

# ------------------------------------------------------------------
# AE Layer：一个 SwiGLU MLP + Pre-Norm 残差
# ------------------------------------------------------------------
class AELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = LlamaMLP(config)
        self.layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

# ------------------------------------------------------------------
#      Encoder：输入 x_{1:K} → q_φ(z|x)的参数 → 采样 z
#      输出 shape: [B, L, 2*latent_size]  (L=T/K)
#      2* 是因为要同时输出 μ 和 log σ
# ------------------------------------------------------------------
class Encoder(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.latent_size = config.latent_size

        # 词嵌入层 |V| -> hidden_size  
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 若干 AE Layer，分成 2 个 stage
        self.encoder_layers = nn.ModuleList([AELayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2

        # 线性层：将 d 维隐藏态映射到 2l 维 ：μ有l维, log σ有l维；最后l维分别采样成l维的z
        self.hidden_to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)

        # 线性层：把 K 个 token 的拼接压回 d 维（stage-0 结束后）
        self.squeeze_layer = nn.Linear(self.patch_size * config.hidden_size, config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    # ---------- 前向：输入 token id 矩阵 ---------------------------
    # input_ids: [B, T]  ->  reshape -> [B*L, K]
    # return   : [B, L, 2*l]
    # ---------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        batch_size, seq_length = input_ids.shape
        num_patches = seq_length // self.patch_size    # L = T/K
        input_ids = input_ids.reshape(batch_size * num_patches, self.patch_size)       #查表得到embedding [B*L, K]

        inputs_embeds = self.embed_tokens(input_ids)    # [B*L, K, d]
        if self.training:
            inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)

        hidden_states = inputs_embeds     # 下文简写 h

        # 3. 两阶段编码
        for stage in range(2):
            for layer_idx in range(self.num_stage_layers):
                encoder_idx = stage * self.num_stage_layers + layer_idx
                encoder_layer = self.encoder_layers[encoder_idx]        # 维度不变
                hidden_states = encoder_layer(hidden_states)    # h (B, hidden_size)

            # stage-0 结束后：把 K 个向量拼起来压回 d 维
            if stage == 0:
                hidden_states = hidden_states.view(batch_size * num_patches, 1, -1)
                hidden_states = self.squeeze_layer(hidden_states)

        hidden_states = self.norm(hidden_states)        # [B*L, d]

        
        latent_states = self.hidden_to_latent(hidden_states)    # [B*L, 2*l]
        latent_states = latent_states.reshape(batch_size, num_patches, self.latent_size * 2)     # [B, L, 2*l]  

        return latent_states    # μ、log σ！

# ------------------------------------------------------------------
# 📄 2.1 Decoder：给定 z → 还原 logits  over  K 个 token
#      输出 shape: [B, L, K, |V|]
# ------------------------------------------------------------------
class Decoder(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.num_stage_layers = config.num_decoder_layers // 2

        # z ∈ R^l  →  h ∈ R^d
        self.latent_to_hidden = nn.Linear(config.latent_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList([AELayer(config) for _ in range(config.num_decoder_layers)])

        self.expand_layer = nn.Linear(config.hidden_size, self.patch_size * config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    # ---------- 前向：输入采样后的 z ---------------------------
    # latent_states: [B, L, l]  (已重参数化)  L是patch的个数
    # return logits: [B, L*K, |V|]
    # ----------------------------------------------------------
    def forward(
        self,
        latent_states,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        batch_size, seq_length, latent_size = latent_states.shape   # seq_length = L
        hidden_states = self.latent_to_hidden(latent_states)     # [B, L, l] -> [B, L, d]

        for stage in range(2):
            for layer_idx in range(self.num_stage_layers):
                decoder_idx = stage * self.num_stage_layers + layer_idx
                decoder_layer = self.decoder_layers[decoder_idx]
                hidden_states = decoder_layer(hidden_states)

            if stage == 0:  # 第一阶段执性完的时候扩维
                hidden_states = self.expand_layer(hidden_states)    # [B, L, K*d]
                hidden_states = hidden_states.reshape(batch_size, seq_length * self.patch_size, -1)     # [B, L*K, d]

        hidden_states = self.norm(hidden_states)

        logits = F.linear(hidden_states, self.lm_head_weight)       # [B, L*K, |V|]，最后一维是词表大小，也就是每个位置上的（未归一化）的概率
        # self.lm_head_weight是权重矩阵（与encoder的嵌入矩阵绑定）

        return logits

# ------------------------------------------------------------------
# 完整 VAE：封装 Encoder + 采样 + Decoder + 损失
#  训练时返回 CE + β·KL
# ------------------------------------------------------------------
class Autoencoder(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.patch_size = config.patch_size
        # 让 decoder 复用 encoder 的嵌入矩阵作为输出层权重
        self.decoder.lm_head_weight = self.encoder.embed_tokens.weight

        # 正则化超参     
        self.ae_dropout = config.ae_dropout
        self.kl_clamp = config.kl_clamp      # λ_KL，KL clipping 阈值
        self.kl_weight = config.kl_weight    # β，KL 权重

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    # ---------- 完整前向：训练模式返回 VAE 总损失 ------------------
    # input_ids: [B, T]  (T 必须是 K 的倍数)
    # labels   : [B, T]  与 input_ids 相同，用于 CE
    # return   : CausalLMOutputWithPast( loss = CE + β·KL )
    # ---------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # 1. 输入 dropout（token 级别）
        input_ids = input_ids.reshape(-1, self.patch_size)   # [B*L, K]
        if self.training:
            mask = torch.rand_like(input_ids.float()) > self.ae_dropout
            input_ids = input_ids * mask.long()      # 随机 mask 部分 token

        # 2. 编码得 (μ, logσ)
        latent_states = self.encoder(input_ids=input_ids)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)
        std = torch.exp(log_std)
        # 采样得到  z ~ q_φ(z|x)
        eps = torch.randn_like(mean)
        latent_states = mean + eps * std
        
        # 3. latent dropout
        latent_states = torch.nn.functional.dropout(latent_states, p=self.ae_dropout, training=self.training)

        # 4. KL(q||N(0,I))  逐维计算后 clamp 再求和      
        kl_loss = 0.5 * (torch.pow(mean, 2) + torch.pow(std, 2) - 1 - log_std * 2)
        kl_loss = torch.clamp(kl_loss, min = self.kl_clamp)
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        # 5. 解码 & CE
        logits = self.decoder(latent_states=latent_states).float()      # [B*L*K, |V|] ，L*K是 token数：K个一组，共L组
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, self.config.vocab_size)
        labels = labels.view(-1).to(logits.device)
        loss = loss_fct(logits, labels) 

        # 总的loss：CE + KL散度，KL是为了约束z的分布使其对齐标准正态分布，分布更平滑，防止微小扰动
        if self.training:
            loss = loss * self.patch_size + kl_loss * self.kl_weight

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )

