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
def prepare_gradient_for_backward(self):
    """
        Prepares the gradient for the backward computation by saving and
        clearing any existing sharded gradient in ``.grad`` to enable computing
        a new unsharded gradient.
        """
    _p_assert(self._training_state in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.IDLE), 'Expects to be in `BACKWARD_PRE` or `IDLE` (if prefetching)')
    flat_param = self.flat_param
    if flat_param.grad is not None and (flat_param.grad.size() != flat_param._unpadded_unsharded_size or flat_param.grad.device != flat_param.device):
        self._check_on_compute_device(self.flat_param)
        grad_offloaded = flat_param.grad.device != self.device
        _p_assert(not grad_offloaded or self._offload_params, f'Expects the sharded gradient to be on {self.device} but got {flat_param.grad.device}')
        prev_iter_synced_gradients = flat_param.grad.size() == flat_param._local_shard.size()
        if prev_iter_synced_gradients:
            if not grad_offloaded:
                flat_param._saved_grad_shard = flat_param.grad.data
                sharded_grad = flat_param._saved_grad_shard
            else:
                _p_assert(hasattr(flat_param, '_cpu_grad'), '`_cpu_grad` should be defined if the gradient is on CPU')
                sharded_grad = flat_param._cpu_grad
            local_shard_dtype = flat_param._local_shard.dtype
            if self._keep_low_precision_grads and sharded_grad.dtype != local_shard_dtype:
                sharded_grad.data = sharded_grad.to(local_shard_dtype)
        else:
            padded_unsharded_size = flat_param._padded_unsharded_size
            _p_assert(flat_param.grad.size() == padded_unsharded_size, f'Expects `.grad` to be the unsharded gradient in `no_sync()` with size {padded_unsharded_size} but got size {flat_param.grad.size()}')
        flat_param.grad = None