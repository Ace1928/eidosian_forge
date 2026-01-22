import inspect
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
def recompute_mean_var(input: Tensor, rstd: Tensor, inner_dim_indices: List[int], keepdim: bool):
    mean = torch.mean(input, dim=inner_dim_indices, keepdim=keepdim)
    var = torch.var(input, dim=inner_dim_indices, unbiased=False, keepdim=keepdim)
    eps = torch.pow(1 / rstd, 2) - var
    eps = eps.detach()
    rstd = 1 / torch.sqrt(var + eps)
    return (mean, rstd)