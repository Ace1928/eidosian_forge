import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def global_token_pattern(attention_query_mask: torch.Tensor) -> torch.Tensor:
    assert attention_query_mask.ndim == 1
    assert attention_query_mask.dtype == torch.bool
    attention_query_mask = attention_query_mask[None, :]
    mask = attention_query_mask | attention_query_mask.transpose(1, 0)
    return mask