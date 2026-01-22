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
def local_metadata_dict(self) -> Dict[str, Any]:
    """
        Get the information needed to reconstruct the model from shards offline.

        See the `consolidate_shard_weights` method below.
        """
    param_metadata = []
    for path, m in self.named_modules():
        if isinstance(m, FullyShardedDataParallel):
            metadata: Dict[str, Any] = {}
            metadata['fsdp_path'] = _clean_path(path)
            metadata['params'] = {}
            metadata['no_broadcast_optim_state'] = m.no_broadcast_optim_state
            shared_param_info = []
            for mpath_dst, mpath_src, _, src_name, _, dst_name in m._shared_param_infos:
                src_param_path = _clean_path(mpath_src + '.' + src_name if mpath_src else src_name)
                dst_param_path = _clean_path(mpath_dst + '.' + dst_name if mpath_dst else dst_name)
                shared_param_info.append((src_param_path, dst_param_path))
            metadata['shared_param_info'] = shared_param_info
            for i, p in enumerate(m.params):
                if i < m._num_flatten_params:
                    backing_param_name = m.module.flat_param_names[i]
                    names, shapes, numels = m.module.metadata(i)
                else:
                    assert len(m._param_name_groups[i]) == 1
                    backing_param_name = m._param_name_groups[i][0]
                    names = [backing_param_name]
                    shapes = [p._orig_size]
                    numels = [p._orig_size.numel()]
                backing_param_name = _clean_path(backing_param_name)
                metadata['params'][backing_param_name] = {'names': [_clean_path(n) for n in names], 'shapes': shapes, 'numels': numels, 'padding': m.numel_padded_per_param[i]}
            param_metadata.append(metadata)
    buffer_names = [_clean_path(buffer_name) for buffer_name, _ in self.named_buffers(recurse=True)]
    return dict(param_metadata=param_metadata, buffer_names=buffer_names)