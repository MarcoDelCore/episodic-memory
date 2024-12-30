import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from model.layers import (
    Embedding,
    VisualProjection,
    FeatureEncoder,
    CQAttention,
    CQConcatenate,
    ConditionedPredictor,
    HighLightLayer,
    BertEmbedding,
)

def build_optimizer_and_scheduler(model, configs):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler

class VSLBase(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLBase, self).__init__()
        self.configs = configs
        self.video_projection = VisualProjection(
            visual_dim=configs.video_feature_dim,
            dim=configs.dim,
            drop_rate=configs.drop_rate,
        )
        self.embedding_net = Embedding(
            num_words=configs.word_size,
            num_chars=configs.char_size,
            out_dim=configs.dim,
            word_dim=configs.word_dim,
            char_dim=configs.char_dim,
            word_vectors=word_vectors,
            drop_rate=configs.drop_rate,
        )
        self.attention_block = MultiHeadAttentionBlock(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
        )
        self.conv_block = DepthwiseSeparableConvBlock(
            dim=configs.dim,
            kernel_size=3,
            drop_rate=configs.drop_rate,
        )
        self.predictor = nn.Linear(configs.dim, 2)
        self.init_parameters()

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_projection(video_features)
        query_features = self.embedding_net(word_ids, char_ids)
        fused_features = torch.cat([video_features, query_features], dim=1)
        fused_features = self.attention_block(fused_features, mask=q_mask)
        fused_features = self.conv_block(fused_features)
        logits = self.predictor(fused_features)
        start_logits, end_logits = logits.split(1, dim=1)
        return start_logits.squeeze(-1), end_logits.squeeze(1)
