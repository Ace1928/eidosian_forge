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
def auto_wrap_bn(module: nn.Module, single_rank_pg: bool=False, process_group: Optional['ProcessGroup']=None, fsdp_config: Optional[Dict[str, Any]]=None, wrap_it: bool=True, assert_on_collision: bool=True) -> nn.Module:
    """
    Auto wrap all BatchNorm (BN) instances with a safer FSDP, esp. when convert
    to sync BN is used and the outer FSDP is flattening.

    We put BN in is own full precision, unflatten, single GPU group FSDP.  Note, SyncBNs still have
    a group size == world_size. The input and output for BN are still FP16 in mixed precision mode.
    See ``keep_batchnorm_fp32`` here: https://nvidia.github.io/apex/amp.html

    This needs to be done at each rank, like models being wrapped by FSDP at each rank.

    Args:
        module (nn.Module):
            The model (or part of the model) in which BN to be pre-wrapped.
        single_rank_pg (bool):
            If true, put BNs in a single-rank process group. Default False.
            This might be needed for Apex sync BN support. Still under construction.
        process_group (ProcessGroup):
            Optional process group to be used.
        fsdp_config (Dict):
            Optional fsdp_config to be used.
        wrap_it (bool):
            Whether or not wrap the module after setting the config.
            Default: True
        assert_on_collision (bool):
            Whether or not assert if a wrapper_config already exists on the module.
            Default: True

    Returns:
        Processed module, where BNs are wrapped with a special FSDP instance.
    """
    pg = process_group
    if single_rank_pg:
        my_rank = dist.get_rank()
        pg = get_process_group_cached(ranks=[my_rank])
    if fsdp_config is None:
        fsdp_config = {'process_group': pg, 'mixed_precision': False, 'flatten_parameters': False, 'reshard_after_forward': False, 'bucket_cap_mb': 0, 'force_input_to_fp32': False}
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if assert_on_collision:
                assert not hasattr(m, 'wrapper_config'), "Module shouldn't already have a wrapper_config. Is it tagged already by another policy?"
            m.wrapper_config = fsdp_config
    with enable_wrap(config_auto_wrap_policy, wrapper_cls=FullyShardedDataParallel) if wrap_it else contextlib.suppress():
        return auto_wrap(module)