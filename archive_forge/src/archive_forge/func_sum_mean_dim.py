import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def sum_mean_dim(self: List[int], opt_dims: Optional[List[int]], keep_dim: bool, dt: Any):
    out: List[int] = []
    if opt_dims is None or len(opt_dims) == 0:
        dims: List[int] = list(range(len(self)))
    else:
        dims = opt_dims
    for idx in range(len(self)):
        is_mean_dim: bool = False
        for reduce_dim in dims:
            if idx == maybe_wrap_dim(reduce_dim, len(self)):
                is_mean_dim = True
        if is_mean_dim:
            if keep_dim:
                out.append(1)
        else:
            out.append(self[idx])
    return out