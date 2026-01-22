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
def pre_unshard(self) -> bool:
    """
        Returns: ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
    if self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS and self._skipped_use_sharded_views:
        self._use_sharded_views()
    ret = False
    if self._use_orig_params and (not self._skip_writeback_check):
        ret = self._writeback_orig_params()
    if self.uses_sharded_strategy and (not self._offload_params) and (not self.needs_unshard()):
        pass
    elif self._uses_param_mixed_precision and (not self._force_full_precision):
        self._use_low_precision_shard()
        ret = True
    elif self._offload_params and self.flat_param.device != self.device:
        self.flat_param_to(self.device, non_blocking=True)
        ret = True
    self._check_on_compute_device(self.flat_param)
    return ret