import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def sdpa_flop_count(query_shape, key_shape, value_shape):
    """
    Count flops for self-attention.

    NB: We can assume that value_shape == key_shape
    """
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    assert b == _b2 == _b3 and h == _h2 == _h3 and (d_q == _d2) and (s_k == _s3) and (d_q == _d2)
    total_flops = 0
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops