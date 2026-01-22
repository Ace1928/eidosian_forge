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
def transpose_block_attn(self, query, key, value, sample):
    block_ctx = self.block_ctx
    batch_size, seq_len, embed_dim = value.shape
    if sample:
        block_len = (seq_len - 1) % block_ctx
        key = key[:, block_len::block_ctx, :]
        value = value[:, block_len::block_ctx, :]
        return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
    else:
        query_length = query.shape[1]
        query = query.view(batch_size, query_length // block_ctx, block_ctx, embed_dim)
        query = query.transpose(1, 2).contiguous()
        query = query.view(batch_size * block_ctx, query_length // block_ctx, embed_dim)
        key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
        key = key.transpose(1, 2).contiguous()
        key = key.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)
        value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
        value = value.transpose(1, 2).contiguous()
        value = value.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)
        block_attn = self.dense_attn(query, key, value, sample)
        block_attn = block_attn.view(batch_size, block_ctx, query_length // block_ctx, embed_dim)
        block_attn = block_attn.transpose(1, 2).contiguous()
        block_attn = block_attn.view(batch_size, query_length, embed_dim)
        return block_attn