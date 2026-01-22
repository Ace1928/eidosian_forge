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
@staticmethod
def consolidate_shard_weights(shard_weights: List[Dict[str, torch.Tensor]], shard_metadata: List[Dict[str, Any]], with_module_buffers: bool=True, strict: bool=True) -> Dict[str, torch.Tensor]:
    """
        Given a list of weights and meta data associated to N shards, reconstruct
        the weights of an equivalent consolidated (non-sharded) state dict.

        Module parameters are consolidated using the shard metadata.

        Module buffers are taken from shard 0: this assumes that module buffers
        are either synchronized or that the shard 0 value is valid for all shards.
        If this behavior is not correct for your module (for instance if buffers
        needs to be all-reduced instead), you can disable it with `with_module_buffers=False`.

        This method is used to re-assemble checkpoints of shards without
        having to instantiate FSDP wrappers with the world size (i.e. large
        number of GPUs) originally used to save the shards.

        Args:
            shard_weights (List[Dict[str, torch.Tensor]]):
                List of dictionaries that contains sharded weights from
                each rank.
            shard_metadata (List[Dict[str, Any]]):
                List of dictionaries that contains metadata from each shard.
                See `local_metadata_dict` above.
            with_module_buffers (bool):
                If shard 0's buffer should be returned in the consolidated
                weight dict.
                Default: True.
            strict (bool):
                allow incomplete shard weights. if True, every key in the metadata must be present in the weights.

        """
    if len(shard_weights) != len(shard_metadata) or not len(shard_weights):
        raise ValueError('Require metadata for each shard and non-empty shards')
    consolidated_weights = {}
    original_world_size = len(shard_weights)
    for fsdp_obj_idx, metadata in enumerate(shard_metadata[0]['param_metadata']):
        fsdp_path = metadata['fsdp_path']
        params = metadata['params']
        for backing_param_name, v in params.items():
            in_state_dict_key = '.'.join([fsdp_path, backing_param_name]) if fsdp_path else backing_param_name
            if in_state_dict_key not in shard_weights[0] and (not strict):
                continue
            shards = []
            for rank in range(original_world_size):
                shard = shard_weights[rank][in_state_dict_key]
                pad = shard_metadata[rank]['param_metadata'][fsdp_obj_idx]['params'][backing_param_name]['padding']
                shards.append(_unpad(shard, pad))
                if metadata['no_broadcast_optim_state']:
                    break
            full_param = torch.cat(shards, dim=0)
            names, shapes, numels, _ = v.values()
            assert sum(numels) == full_param.size(0)
            for n, t, s in zip(names, full_param.split(numels), shapes):
                out_state_dict_key = '.'.join([fsdp_path, n]) if fsdp_path else n
                consolidated_weights[out_state_dict_key] = t.view(s)
    for src_path, dest_path in metadata['shared_param_info']:
        consolidated_weights[dest_path] = consolidated_weights[src_path]
    if with_module_buffers:
        for buffer_name in shard_metadata[0]['buffer_names']:
            if buffer_name not in shard_weights[0] and (not strict):
                continue
            consolidated_weights[buffer_name] = shard_weights[0][buffer_name]
    return consolidated_weights