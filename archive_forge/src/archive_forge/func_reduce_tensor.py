import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def reduce_tensor(tensor):
    if tensor.requires_grad and (not tensor.is_leaf):
        raise RuntimeError('Cowardly refusing to serialize non-leaf tensor which requires_grad, since autograd does not support crossing process boundaries.  If you just want to transfer the data, call detach() on the tensor before serializing (e.g., putting it on the queue).')
    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)
    from torch.nested._internal.nested_tensor import NestedTensor
    if tensor.is_nested and (not isinstance(tensor, NestedTensor)):
        return reduce_nested_tensor(tensor)
    if tensor.layout in {torch.sparse_coo, torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc}:
        return reduce_sparse_tensor(tensor)
    storage = tensor._typed_storage()
    if storage._untyped_storage.device.type == 'cuda':
        device, handle, storage_size_bytes, storage_offset_bytes, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        return (rebuild_cuda_tensor, (type(tensor), tensor.size(), tensor.stride(), tensor_offset, type(storage), tensor.dtype, device, handle, storage_size_bytes, storage_offset_bytes, tensor.requires_grad, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required))
    metadata = (tensor.storage_offset(), tensor.size(), tensor.stride(), tensor.requires_grad)
    return (rebuild_tensor, (type(tensor), storage, metadata))