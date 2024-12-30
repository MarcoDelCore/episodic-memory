"""VSLBase Baseline for Ego4D Episodic Memory -- Natural Language Queries.
"""
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from model.layers import (
    Embedding,
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
        # Proiezione video
        self.video_affine = nn.Linear(configs.video_feature_dim, configs.dim)
        # Embedding testuale semplice
        self.embedding_net = Embedding(
            num_words=configs.word_size,
            num_chars=configs.char_size,
            out_dim=configs.dim,
            word_dim=configs.word_dim,
            char_dim=configs.char_dim,
            word_vectors=word_vectors,
            drop_rate=configs.drop_rate,
        )
        # Modulo di fusione semplice
        self.fusion_layer = nn.Linear(2 * configs.dim, configs.dim)
        # Predittore per inizio e fine
        self.predictor = nn.Linear(configs.dim, 2)
        # Inizializzazione dei pesi
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        # Proiezione video
        video_features = self.video_affine(video_features)
        # Embedding testuale
        query_features = self.embedding_net(word_ids, char_ids)
        # Fusione (concatenazione + feed-forward)
        fused_features = torch.cat([video_features, query_features], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        # Predizione inizio e fine
        logits = self.predictor(fused_features)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        loss_fn = nn.CrossEntropyLoss()
        start_loss = loss_fn(start_logits, start_labels)
        end_loss = loss_fn(end_logits, end_labels)
        return start_loss + end_loss
