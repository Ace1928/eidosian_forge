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
@no_type_check
@torch.no_grad()
def init_flat_param_attributes(self) -> None:
    """
        This initializes some attributes on the handle's ``FlatParameter``.
        This should be called during lazy initialization since it requires the
        parameter to be on the compute device if not offloading to CPU and we
        want to give users the chance to move the parameter appropriately after
        the FSDP constructor.

        For each tensor attribute on the ``FlatParameter``, see the unshard and
        reshard methods in this class for the allocation and free pattern.
        """
    flat_param = self.flat_param
    if flat_param.dtype != self._orig_param_dtype:
        if not self._low_prec_param_dtype_specified:
            self._fwd_bwd_param_dtype = flat_param.dtype
        if not self._low_prec_reduce_dtype_specified and (not self._low_prec_param_dtype_specified):
            self._reduce_dtype = flat_param.dtype
        self._orig_param_dtype = flat_param.dtype
    cpu_device = torch.device('cpu')
    if self._offload_params:
        _p_assert(flat_param.device == cpu_device, f'Expects the `FlatParameter` to be on CPU when parameter CPU offloading is enabled, not {flat_param.device}')
    else:
        self._check_on_compute_device(self.flat_param)
    flat_param._local_shard = flat_param.data
    if self._offload_params:
        flat_param._local_shard = flat_param._local_shard.pin_memory()
        flat_param._cpu_grad = torch.zeros_like(flat_param._local_shard, device=cpu_device).pin_memory()
    if self._uses_param_mixed_precision:
        flat_param._mp_shard = torch.empty_like(flat_param._local_shard, device=self.device, dtype=self._fwd_bwd_param_dtype)
        _free_storage(flat_param._mp_shard)
    if self.uses_sharded_strategy:
        unsharded_param_dtype = self._fwd_bwd_param_dtype if self._uses_param_mixed_precision else flat_param.dtype
        padded_unsharded_numel = flat_param.numel() * self.world_size
        flat_param._full_param_padded = torch.empty(padded_unsharded_numel, device=self.device, dtype=unsharded_param_dtype)
        flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
        _free_storage(flat_param._full_param_padded)
        if self._uses_param_mixed_precision:
            flat_param._full_prec_full_param_padded = torch.empty(padded_unsharded_numel, device=self.device, dtype=flat_param.dtype)
            _free_storage(flat_param._full_prec_full_param_padded)