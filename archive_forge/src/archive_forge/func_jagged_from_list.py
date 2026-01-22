from typing import Tuple
import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403
def jagged_from_list(tensors: List[torch.Tensor], offsets: Optional[torch.Tensor], dtype=None, device=None) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""
    if not len(set((t.dtype for t in tensors))) == 1:
        raise RuntimeError('When constructing a nested tensor, all tensors in list must have the same dtype')
    if not len(set((t.device for t in tensors))) == 1:
        raise RuntimeError('When constructing a nested tensor, all tensors in list must be on the same device')
    sizes = [t.shape for t in tensors]
    non_first_sizes = [s[1:] for s in sizes]
    at_most_first_ragged = all((s == non_first_sizes[0] for s in non_first_sizes))
    if not at_most_first_ragged:
        raise RuntimeError('Cannot represent given tensor list as a nested tensor with the jagged layout. Note that the jagged layout only represents shapes of the form (B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged.')
    values = torch.cat(tensors, dim=0)
    to_kwargs = {}
    if device is not None:
        to_kwargs['device'] = device
    if dtype is not None:
        to_kwargs['dtype'] = dtype
    values = values.to(**to_kwargs)
    if offsets is None:
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=values.device), torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0)])
    max_seqlen = max([t.shape[0] for t in tensors])
    min_seqlen = min([t.shape[0] for t in tensors])
    return (ViewNestedFromBuffer.apply(values, offsets, max_seqlen, min_seqlen), offsets)