from functools import update_wrapper
from numbers import Number
from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch.overrides import is_tensor_like
def vec_to_tril_matrix(vec: torch.Tensor, diag: int=0) -> torch.Tensor:
    """
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    n = (-(1 + 2 * diag) + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and round(n) - n > eps:
        raise ValueError(f'The size of last dimension is {vec.shape[-1]} which cannot be expressed as ' + 'the lower triangular part of a square D x D matrix.')
    n = round(n.item()) if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat