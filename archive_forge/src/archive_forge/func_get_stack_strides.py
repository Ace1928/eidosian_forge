from typing import List, Optional, Sequence, Tuple, Union
import torch
from .common import _get_storage_base
def get_stack_strides(tensors: Sequence[torch.Tensor], dim: int) -> Optional[Tuple[int, ...]]:
    """
    If the tensors are already stacked on dimension :code:`dim`,         returns the strides of the stacked tensors.         Otherwise returns :code:`None`.
    """
    if len(tensors) <= 1 or dim > tensors[0].ndim:
        return None
    final_stride = []
    for i in range(tensors[0].ndim + 1):
        if i == dim:
            final_stride.append(tensors[1].storage_offset() - tensors[0].storage_offset())
            continue
        if i > dim:
            i -= 1
        final_stride.append(tensors[0].stride(i))
    storage_data_ptr: Optional[int] = None
    for i, x in enumerate(tensors[1:]):
        if x.shape != tensors[0].shape:
            return None
        if x.stride() != tensors[0].stride():
            return None
        if x.storage_offset() != tensors[0].storage_offset() + (i + 1) * final_stride[dim]:
            return None
        if storage_data_ptr is None:
            storage_data_ptr = _get_storage_base(tensors[0])
        if _get_storage_base(x) != storage_data_ptr:
            return None
    return tuple(final_stride)