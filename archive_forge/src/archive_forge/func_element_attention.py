import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def element_attention(self, query, key, padding_mask, causal_mask):
    """
        Apply element-wise attention via relu^2 or laplace. Same as original implementation but with standardized
        causal attention mask. Expects the Hugging Face standard attention mask paradigm: 1 for not masked, and 0 for
        masked.
        """
    seq_len = key.size(2)
    if padding_mask is not None:
        lengths = padding_mask.sum(-1, keepdim=True)
        lengths = lengths.clamp(min=1.0).unsqueeze(-1)
    else:
        lengths = seq_len
    if causal_mask is not None:
        lengths = causal_mask.sum(dim=-1, keepdim=True)
    bias = self.rel_pos_bias(seq_len)
    if seq_len != query.size(2):
        if query.size(2) != 1:
            raise ValueError('Size mismatch between Q and K in element attention')
        bias = bias[-1:]
    qk = torch.matmul(query, key.transpose(2, 3)) / lengths + bias
    attn_weights = ACT2FN[self.config.attention_activation](qk).type_as(qk)
    if padding_mask is not None:
        attn_weights = attn_weights * padding_mask.unsqueeze(2)
    if causal_mask is not None:
        attn_weights = attn_weights * causal_mask
    return attn_weights