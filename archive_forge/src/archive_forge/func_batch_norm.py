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
def batch_norm(input: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor], weight: Optional[Tensor]=None, bias: Optional[Tensor]=None, training: bool=False, momentum: float=0.1, eps: float=1e-05) -> Tensor:
    """Apply Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    if has_torch_function_variadic(input, running_mean, running_var, weight, bias):
        return handle_torch_function(batch_norm, (input, running_mean, running_var, weight, bias), input, running_mean, running_var, weight=weight, bias=bias, training=training, momentum=momentum, eps=eps)
    if training:
        _verify_batch_size(input.size())
    return torch.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled)