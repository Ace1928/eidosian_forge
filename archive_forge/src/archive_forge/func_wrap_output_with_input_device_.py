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
def wrap_output_with_input_device_(x, common_device):
    if common_device is not None and x.device.type == 'meta':
        from torch._subclasses.fake_tensor import FakeTensorMode
        fake_mode = FakeTensorMode()
        fake_mode.in_kernel_invocation = True
        converter = fake_mode.fake_tensor_converter
        return converter.from_meta_and_device(fake_mode, x, common_device)
    return x