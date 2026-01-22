import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
@aten.hardshrink.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.hardshrink)
@out_wrapper()
def hardshrink(a: TensorLikeType, lambd: float=0.5):
    return torch.where(torch.abs(a) <= lambd, 0, a)