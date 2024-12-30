"""
Layers to construct the VSLBase model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

class Embedding(nn.Module):
    def __init__(
        self,
        num_words,
        num_chars,
        word_dim,
        char_dim,
        drop_rate,
        out_dim,
        word_vectors=None,
    ):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(
            num_words, word_dim, drop_rate, word_vectors=word_vectors
        )
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate)
        # output linear layer
        self.linear = Conv1D(
            in_dim=word_dim + 100,
            out_dim=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, 100)
        emb = torch.cat(
            [word_emb, char_emb], dim=2
        )  # (batch_size, w_seq_len, word_dim + 100)
        emb = self.linear(emb)  # (batch_size, w_seq_len, dim)
        return emb