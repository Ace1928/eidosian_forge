from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::get_pool_ceil_padding')
@_beartype.beartype
def get_pool_ceil_padding(input, kernel_size, stride, padding):
    sizes = symbolic_helper._get_tensor_sizes(input)
    dim = sizes[-len(padding):] if sizes is not None else None
    if dim is None or any((i is None for i in dim)):
        return symbolic_helper._unimplemented('get_pool_ceil_padding', 'input size not accessible', input)
    ceiled_output_dim = [int(math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i]))) + 1 for i in range(0, len(padding))]
    ceiled_output_dim = [ceiled_output_dim[i] - 1 if (ceiled_output_dim[i] - 1) * stride[i] >= dim[i] + padding[i] else ceiled_output_dim[i] for i in range(0, len(ceiled_output_dim))]
    padding_ceil = [0 if stride[i] == 1 else kernel_size[i] - (dim[i] + 2 * padding[i] - ((ceiled_output_dim[i] - 1) * stride[i] + 1)) for i in range(0, len(padding))]
    padding_ceil = [(int(padding_ceil[i]) if padding_ceil[i] < kernel_size[i] - 1 else int(kernel_size[i] - 1)) if padding_ceil[i] + 2 * padding[i] >= kernel_size[i] else int(padding_ceil[i]) for i in range(0, len(padding_ceil))]
    return padding_ceil