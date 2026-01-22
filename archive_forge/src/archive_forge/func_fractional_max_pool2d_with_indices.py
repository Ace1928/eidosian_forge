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
def fractional_max_pool2d_with_indices(input: Tensor, kernel_size: BroadcastingList2[int], output_size: Optional[BroadcastingList2[int]]=None, output_ratio: Optional[BroadcastingList2[float]]=None, return_indices: bool=False, _random_samples: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
    """
    fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH \\times kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k \\times k`)
                     or a tuple `(kH, kW)`
        output_size: the target output size of the image of the form :math:`oH \\times oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \\times oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if has_torch_function_variadic(input, _random_samples):
        return handle_torch_function(fractional_max_pool2d_with_indices, (input, _random_samples), input, kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    if output_size is None and output_ratio is None:
        raise ValueError('fractional_max_pool2d requires specifying either an output_size or an output_ratio')
    if output_size is None:
        assert output_ratio is not None
        if len(output_ratio) > 2:
            raise ValueError('fractional_max_pool2d requires output_ratio to either be a single Int or tuple of Ints.')
        _output_ratio = _pair(output_ratio)
        output_size = [int(input.size(-2) * _output_ratio[0]), int(input.size(-1) * _output_ratio[1])]
    if _random_samples is None:
        n_batch = 1 if input.dim() == 3 else input.size(0)
        _random_samples = torch.rand(n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)