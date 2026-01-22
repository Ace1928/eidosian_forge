import contextlib
import copy
from enum import Enum, auto
import functools
import logging
from math import inf
import os
import time
import traceback
import typing
from typing import (
import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairscale.internal.containers import apply_to_tensors
from fairscale.internal.parallel import (
from fairscale.internal.params import calc_grad_norm, recursive_copy_to_device
from fairscale.internal.reduce_scatter_bucketer import ReduceScatterBucketer
from fairscale.internal.state_dict import replace_by_prefix_
from fairscale.nn.misc import FlattenParamsWrapper, _enable_pre_load_state_dict_hook
from fairscale.nn.wrap import auto_wrap, config_auto_wrap_policy, enable_wrap
from . import fsdp_optim_utils as ou
def get_shard_from_optim_state_dict(self, full_optim_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Get the portion of the optimizer state dict associated with the shard

        This can be used to get the right sharded optimizer state to be loaded
        into the sharded optimizer for this FSDP rank.

        ..warning:: The input state dict is modified in-place assuming the original
                    full state isn't going to be used anymore. This is done so that
                    we don't need to copy extra state in it. It is caller's responsibility
                    to make copies if it doesn't want the original state dict modified.

        Args:
            full_optim_state_dict (dict):
                consolidated optimizer state returned by ``gather_full_optim_state``,
                or loaded from a checkpoint.

        Returns:
            (dict): a shard of the optimizer state.
        """
    instance_list = get_fsdp_instances(self, skip_empty=True)
    ou.check_param_counts_before_sharding(full_optim_state_dict, len(instance_list))
    ids_not_to_shard = copy.deepcopy(full_optim_state_dict['uncollected_local_ids'])
    if self.flatten_parameters:
        full_optim_state_dict = ou.flatten_optim_state_dict(full_optim_state_dict)
        assert len(full_optim_state_dict['state']) <= len(instance_list), f'{len(full_optim_state_dict['state'])}, {len(instance_list)}'
    for _id, s in full_optim_state_dict['state'].items():
        for k, v in s.items():
            if torch.is_tensor(v) and _id not in ids_not_to_shard:
                v_shard, _ = self._get_shard(v)
            elif isinstance(v, list) and ou.is_singleton_tensor(v[0]):
                v_shard = v[0] if self.rank >= len(v) else v[self.rank]
                assert ou.is_singleton_tensor(v_shard)
            else:
                v_shard = v
            full_optim_state_dict['state'][_id][k] = v_shard
    return full_optim_state_dict