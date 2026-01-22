from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def triplet_margin_with_distance_loss(anchor: Tensor, positive: Tensor, negative: Tensor, *, distance_function: Optional[Callable[[Tensor, Tensor], Tensor]]=None, margin: float=1.0, swap: bool=False, reduction: str='mean') -> Tensor:
    """Compute the triplet margin loss for input tensors using a custom distance function.

    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError('F.triplet_margin_with_distance_loss does not support JIT scripting: functions requiring Callables cannot be scripted.')
    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(triplet_margin_with_distance_loss, (anchor, positive, negative), anchor, positive, negative, distance_function=distance_function, margin=margin, swap=swap, reduction=reduction)
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError(f'{reduction} is not a valid value for reduction')
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    if not (a_dim == p_dim and p_dim == n_dim):
        raise RuntimeError(f'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor {a_dim}D, positive {p_dim}D, and negative {n_dim}D inputs')
    if distance_function is None:
        distance_function = torch.pairwise_distance
    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)
    if reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'mean':
        return torch.mean(loss)
    else:
        return loss