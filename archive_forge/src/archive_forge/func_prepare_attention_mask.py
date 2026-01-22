import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
def prepare_attention_mask(self, hidden_states, attention_mask):
    batch_size, seq_length = hidden_states.shape[:2]
    causal_mask = torch.full((seq_length, seq_length), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device)
    causal_mask = torch.triu(causal_mask, 1)
    extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand((batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape)
    if attention_mask is not None:
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(self.dtype).min
        extended_attention_mask = extended_causal_mask + extended_attention_mask
    else:
        extended_attention_mask = extended_causal_mask
    return extended_attention_mask.to(hidden_states.dtype)