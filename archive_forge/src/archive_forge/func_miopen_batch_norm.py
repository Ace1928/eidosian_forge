import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(input: torch.Tensor, weight: torch.Tensor, bias: typing.Optional[torch.Tensor], running_mean: typing.Optional[torch.Tensor], running_var: typing.Optional[torch.Tensor], training: bool, exponential_average_factor: float, epsilon: float):
    a, b, c = aten.native_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    if training:
        return (a, b, c)
    return (a, weight.new_zeros((0,)), weight.new_zeros((0,)))