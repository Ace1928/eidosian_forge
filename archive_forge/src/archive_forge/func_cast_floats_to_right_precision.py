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
def cast_floats_to_right_precision(to_fp16: bool, no_grad: bool, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """
    Cast floating point Tensors in *args or **kwargs to FP16 or FP32 if they are not.
    We also retain the requires_grad flag so that casting doesn't affect the autograd graph.
    """

    def fn_fp16(x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.float32:
            y = x.half()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x

    def fn_fp32(x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.float16:
            y = x.float()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x
    fn = fn_fp16 if to_fp16 else fn_fp32
    context = torch.no_grad() if no_grad else contextlib.suppress()
    with context:
        return (apply_to_tensors(fn, args), apply_to_tensors(fn, kwargs))