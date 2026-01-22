import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None):
    emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
    if hasattr(self, 'weights'):
        emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
    self.weights = nn.Parameter(emb_weights)
    self.weights.requires_grad = False
    self.weights.detach_()