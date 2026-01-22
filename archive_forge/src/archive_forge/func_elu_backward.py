import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition(aten.elu_backward)
@out_wrapper('grad_input')
@pw_cast_for_opmath
def elu_backward(grad_output: Tensor, alpha: float, scale: float, input_scale: float, is_result: bool, self_or_result: Tensor):
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    if is_result:
        return torch.where(self_or_result <= 0, grad_output * negiptcoef * (self_or_result + negcoef), grad_output * poscoef)
    else:
        return torch.where(self_or_result <= 0, grad_output * negiptcoef * negcoef * torch.exp(self_or_result * negiptcoef), grad_output * poscoef)