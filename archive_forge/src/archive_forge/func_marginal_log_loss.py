import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
def marginal_log_loss(logits, is_correct):
    """Loss based on the negative marginal log-likelihood."""

    def mask_to_score(mask, dtype=torch.float32):
        return (1.0 - mask.type(dtype)) * torch.finfo(dtype).min
    log_numerator = torch.logsumexp(logits + mask_to_score(is_correct, dtype=logits.dtype), dim=-1)
    log_denominator = torch.logsumexp(logits, dim=-1)
    return log_denominator - log_numerator