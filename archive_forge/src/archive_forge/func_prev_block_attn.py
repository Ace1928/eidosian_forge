import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def prev_block_attn(self, query, key, value, sample):
    block_ctx = self.block_ctx
    batch_size, seq_len, embed_dim = value.shape
    if sample:
        block = (seq_len - 1) // block_ctx
        prev_l = (block - 1) * block_ctx
        if block > 0:
            key = key[:, prev_l:prev_l + block_ctx, :]
            value = value[:, prev_l:prev_l + block_ctx, :]
        else:
            key = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
            value = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
        return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
    else:
        query_length = query.shape[1]
        query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
        key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
        key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0))
        key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
        value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
        value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0))
        value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
        if query_length < seq_len:
            nb_query_blocks = query_length // block_ctx
            nb_key_blocks = seq_len // block_ctx
            seq_len = query_length
            key = key.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
            key = key.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)
            value = value.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
            value = value.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)
        return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)