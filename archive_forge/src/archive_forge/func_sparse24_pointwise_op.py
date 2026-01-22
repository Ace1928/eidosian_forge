import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def sparse24_pointwise_op(func, types, args=(), kwargs=None, allow_sparsify_args_list=()):
    self = None
    for tensor in args:
        if isinstance(tensor, Sparse24Tensor):
            self = tensor
    assert self is not None
    args_updated = []
    for i, tensor in enumerate(args):
        if isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, Sparse24Tensor):
                if i in allow_sparsify_args_list:
                    tensor = sparsify24_like(tensor, self)
                else:
                    raise ValueError(f'Operation {func.__module__}.{func.__name__} on Sparse24Tensor requires all operands to be Sparse24Tensors, but operand {i} is a {type(tensor)}')
            if tensor.threads_masks is None or self.threads_masks is None or tensor.threads_masks.data_ptr() != self.threads_masks.data_ptr() or (tensor.threads_masks.stride() != self.threads_masks.stride()):
                raise ValueError(f'Operation {func.__module__}.{func.__name__} on Sparse24Tensor requires all operands to be Sparse24Tensors with the same sparsity pattern')
        args_updated.append(tensor)
    assert isinstance(self, Sparse24TensorCutlass), 'Only implemented for CUTLASS tensors'
    return Sparse24TensorCutlass(self.shape, func(*[x.packed if isinstance(x, Sparse24Tensor) else x for x in args_updated]), self.meta, func(*[x.packed_t if isinstance(x, Sparse24Tensor) else x for x in args_updated]), self.meta_t, self.threads_masks)