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
def shard_metadata(self) -> FlatParamShardMetadata:
    """
        Returns shard-related metadata specific to this rank's shard of the
        flat parameter.
        NOTE: The returned tuple does not include elements for alignment
        padding but does account for the padding.
        """
    fqns_list = []
    shapes_list = []
    numels_list = []
    shard_param_offsets = []
    for fqn, shape, numel, shard_param_info in zip(self.flat_param._fqns, self.flat_param._shapes, self.flat_param._numels, self.flat_param._shard_param_infos):
        if not shard_param_info.in_shard:
            continue
        fqns_list.append(fqn)
        shapes_list.append(shape)
        numels_list.append(numel)
        shard_param_offsets.append((shard_param_info.intra_param_start_idx, shard_param_info.intra_param_end_idx))
    return FlatParamShardMetadata(tuple(fqns_list), tuple(shapes_list), tuple(numels_list), shard_param_offsets)