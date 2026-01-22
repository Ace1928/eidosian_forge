import torch
from .core import _map_mt_args_kwargs, _masks_match, _tensors_match, _wrap_result, is_masked_tensor
def binary_fn(*args, **kwargs):
    return _binary_helper(fn, args, kwargs, inplace=True)