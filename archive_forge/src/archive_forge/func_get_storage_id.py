import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Tuple
from ._base import FILENAME_PATTERN, MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory
def get_storage_id(tensor: 'torch.Tensor') -> Tuple['torch.device', int, int]:
    """
    Return unique identifier to a tensor storage.

    Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    Taken from https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/pytorch_utils.py#L278.
    """
    if tensor.device.type == 'xla' and is_torch_tpu_available():
        import torch_xla
        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)
    return (tensor.device, unique_id, get_storage_size(tensor))