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
@register_decomposition(aten.cudnn_batch_norm_backward)
@out_wrapper('out0', 'out1', 'out2')
def cudnn_batch_norm_backward(input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_var: Optional[Tensor], epsilon: float, reserveSpace: Tensor):
    return aten.native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_var, True, epsilon, [True, True, True])