import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
def prepare_gradient_for_optim(self):
    """
        Prepares the gradient for optimizer computation by moving the sharded
        gradient to the ``.grad`` attribute.
        """

    def cast_grad_to_param_dtype_if_needed(flat_param):
        if not self._force_full_precision and self._keep_low_precision_grads:
            _p_assert(flat_param.grad is not None, 'Unexpected None grad!')
            if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                if self._use_orig_params:
                    self._use_sharded_grad_views()
    flat_param = self.flat_param
    if hasattr(flat_param, '_cpu_grad'):
        self._check_sharded(flat_param)
        self._check_on_cpu(flat_param)
        flat_param.grad = flat_param._cpu_grad
        cast_grad_to_param_dtype_if_needed(flat_param)
    elif hasattr(flat_param, '_saved_grad_shard'):
        self._check_sharded(flat_param)
        self._check_on_compute_device(flat_param)
        if flat_param._saved_grad_shard is not None:
            self._check_on_compute_device(flat_param._saved_grad_shard)
        if flat_param._post_backward_called:
            flat_param.grad = flat_param._saved_grad_shard
            if flat_param.grad is not None:
                cast_grad_to_param_dtype_if_needed(flat_param)
    else:
        _p_assert(not self.uses_sharded_strategy or not flat_param._post_backward_called, 'All sharded parameters that received a gradient in the post-backward should use `_saved_grad_shard`')
    if hasattr(flat_param, '_saved_grad_shard'):
        delattr(flat_param, '_saved_grad_shard')