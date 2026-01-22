import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def scaled_query_key_softmax(q: torch.Tensor, k: torch.Tensor, att_mask: Optional[Union[AttentionMask, 'SparseCS', torch.Tensor]]) -> torch.Tensor:
    q = q / math.sqrt(k.size(-1))
    if att_mask is not None and isinstance(att_mask, AttentionMask):
        mask: Optional[Union[SparseCS, torch.Tensor]] = att_mask.values
    else:
        mask = att_mask
    att = _matmul_with_mask(q, k.transpose(-2, -1), mask)
    is_causal = isinstance(att_mask, AttentionMask) and att_mask.is_causal
    att = _softmax(att, causal=is_causal)
    return att