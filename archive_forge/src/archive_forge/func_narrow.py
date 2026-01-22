from typing import List, Optional, Union
import torch
from torch import SymInt, Tensor
from torch._C import _add_docstr, _nested  # type: ignore[attr-defined]
from torch.types import _device as Device, _dtype as DType
def narrow(tensor: Tensor, dim: int, start: Union[int, Tensor], length: Union[int, Tensor], layout=torch.strided) -> Tensor:
    """
Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows
similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested tensor
shows only the elements in the interval `[start, start+length)`. As nested representations
allow for a different `start` and `length` at each 'row' of that dimension, :attr:`start` and :attr:`length`
can also be tensors of shape `tensor.shape[0]`.

There's some differences depending on the layout you use for the nested tensor. If using strided layout,
torch.narrow will do a copy of the narrowed data into a contiguous NT with strided layout, while
jagged layout narrow() will create a non-contiguous view of your original strided tensor. This particular
representation is really useful for representing kv-caches in Transformer models, as specialized
SDPA kernels can deal with format easily, resulting in performance improvements.


Args:
    tensor (:class:`torch.Tensor`): a strided tensor, which will be used as the underlying data
        for the nested tensor if using the jagged layout or will be copied for the strided layout.
    dim (int): the dimension where narrow will be applied. Only `dim=1` is supported for the
        jagged layout, while strided supports all dim
    start (Union[int, :class:`torch.Tensor`]): starting element for the narrow operation
    length (Union[int, :class:`torch.Tensor`]): number of elements taken during the narrow op

Keyword arguments:
    layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
        Only strided and jagged layouts are supported. Default: if None, the strided layout.

Example::

    >>> starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    >>> lengths = torch.tensor([3, 2, 2, 1, 5], dtype=torch.int64)
    >>> narrow_base = torch.randn(5, 10, 20)
    >>> nt_narrowed = torch.nested.narrow(narrow_base, 1, starts, lengths, layout=torch.jagged)
    >>> nt_narrowed.is_contiguous()
    False
    """
    if not isinstance(start, (int, SymInt, Tensor)):
        raise RuntimeError('start must be an integer or a tensor')
    if not isinstance(length, (int, SymInt, Tensor)):
        raise RuntimeError('length must be an integer or a tensor')
    if layout == torch.strided:
        if isinstance(start, Tensor) or isinstance(length, Tensor):
            raise RuntimeError('start and length must be integers for the strided layout NT impl')
        nt = as_nested_tensor(torch.unbind(tensor), layout=torch.strided).narrow(dim, start, length)
    elif layout == torch.jagged:
        if dim != 1:
            raise RuntimeError('jagged layout only supports dim=1')
        from torch.nested._internal.nested_tensor import jagged_from_tensor_and_lengths
        if isinstance(start, (int, SymInt)):
            start = torch.tensor([start], device=tensor.device, dtype=torch.int64)
        if isinstance(length, (int, SymInt)):
            length = torch.tensor([length], device=tensor.device, dtype=torch.int64)
        nt, _, _ = jagged_from_tensor_and_lengths(tensor, start, length)
    else:
        raise RuntimeError(f'Specified layout is unsupported for nested narrow: {layout}')
    return nt