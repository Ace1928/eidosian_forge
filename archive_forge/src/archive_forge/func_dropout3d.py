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
def dropout3d(input: Tensor, p: float=0.5, training: bool=True, inplace: bool=False) -> Tensor:
    """Randomly zero out entire channels (a channel is a 3D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(dropout3d, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError(f'dropout probability has to be between 0 and 1, but got {p}')
    inp_dim = input.dim()
    if inp_dim not in (4, 5):
        warn_msg = f'dropout3d: Received a {inp_dim}-D input to dropout3d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout3d exists to provide channel-wise dropout on inputs with 3 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs).'
        warnings.warn(warn_msg)
    is_batched = inp_dim == 5
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)
    result = _VF.feature_dropout_(input, p, training) if inplace else _VF.feature_dropout(input, p, training)
    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)
    return result