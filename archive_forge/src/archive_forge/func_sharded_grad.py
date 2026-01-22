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
@property
def sharded_grad(self) -> Optional[Tensor]:
    """Returns the handle's sharded gradient."""
    flat_param = self.flat_param
    grad: Optional[Tensor]
    if hasattr(flat_param, '_cpu_grad'):
        grad = flat_param._cpu_grad
    elif hasattr(flat_param, '_saved_grad_shard'):
        grad = flat_param._saved_grad_shard
    else:
        _p_assert(flat_param.grad is None or not self.uses_sharded_strategy or self._training_state in (HandleTrainingState.FORWARD, HandleTrainingState.IDLE), 'Sharded strategies should use `_cpu_grad` or `_saved_grad_shard` unless in IDLE or FORWARD')
        grad = flat_param.grad
    return grad