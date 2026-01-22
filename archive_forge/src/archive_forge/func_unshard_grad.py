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
@torch.no_grad()
def unshard_grad(self):
    """
        Unshards the handle's ``FlatParameter`` 's gradient. If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
    if not self.uses_sharded_strategy:
        self._use_unsharded_grad_views()
        return
    flat_param = self.flat_param
    self._check_unsharded(flat_param)
    num_grad_none = torch.zeros(1, dtype=torch.int32, device=self.device)
    num_grad_none[0] = flat_param.grad is None
    dist.all_reduce(num_grad_none, group=self.process_group)
    if num_grad_none[0] == self.world_size:
        flat_param._saved_grad_shard = None
        self._use_unsharded_grad_views()
        return
    padded_unsharded_grad = torch.empty(flat_param._padded_unsharded_size, device=self.device)
    if flat_param.grad is None:
        if self._debug_level == dist.DebugLevel.INFO:
            warnings.warn(f"[Rank {self.rank}] Only some but not all ranks have a `None` `FlatParameter` gradient, so FSDP is using zeros to approximate those ranks' sharded gradients being `None`")
        flat_param._saved_grad_shard = None
        sharded_grad = torch.zeros(flat_param._sharded_size, device=self.device)
    else:
        self._check_sharded(flat_param.grad)
        flat_param._saved_grad_shard = flat_param.grad
        sharded_grad = flat_param._saved_grad_shard
    dist.all_gather_into_tensor(padded_unsharded_grad, sharded_grad, self.process_group)
    unsharded_size = self.flat_param._unpadded_unsharded_size
    flat_param.grad = padded_unsharded_grad[:unsharded_size.numel()].view(unsharded_size)
    self._use_unsharded_grad_views()