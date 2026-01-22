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
def update_p_data(custom_output_tensor: Optional[torch.Tensor]=None) -> None:
    """
            Helper function to update p.data pointer.

            Args:
                custom_output_tensor (torch.Tensor, Optional): if not None, this
                tensor contains the data we just gathered.
            """
    if custom_output_tensor is not None:
        assert p._is_sharded
        p.data = custom_output_tensor
        output_tensors.append((p.data, True))
    elif not p._is_sharded:
        if (self.mixed_precision or self.move_params_to_cpu) and (not force_full_precision):
            assert p._fp16_shard is not None
            p.data = p._fp16_shard
            output_tensors.append((p.data, True))
        else:
            output_tensors.append((p.data, False))
    else:
        p.data = p._full_param_padded
        output_tensors.append((p.data, True))
    p.data = p.data[:p._orig_size.numel()].view(p._orig_size)