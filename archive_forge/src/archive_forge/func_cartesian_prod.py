from typing import (
import operator
import itertools
import torch
from torch._C import _add_docstr
import torch.nn.functional as F
from ._lowrank import svd_lowrank, pca_lowrank
from .overrides import (
from ._jit_internal import boolean_dispatch
from ._jit_internal import _overload as overload
from torch import _VF
def cartesian_prod(*tensors: Tensor) -> Tensor:
    """Do cartesian product of the given sequence of tensors. The behavior is similar to
    python's `itertools.product`.

    Args:
        *tensors: any number of 1 dimensional tensors.

    Returns:
        Tensor: A tensor equivalent to converting all the input tensors into lists,
        do `itertools.product` on these lists, and finally convert the resulting list
        into tensor.

    Example::

        >>> import itertools
        >>> a = [1, 2, 3]
        >>> b = [4, 5]
        >>> list(itertools.product(a, b))
        [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
        >>> tensor_a = torch.tensor(a)
        >>> tensor_b = torch.tensor(b)
        >>> torch.cartesian_prod(tensor_a, tensor_b)
        tensor([[1, 4],
                [1, 5],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5]])
    """
    if has_torch_function(tensors):
        return handle_torch_function(cartesian_prod, tensors, *tensors)
    return _VF.cartesian_prod(tensors)