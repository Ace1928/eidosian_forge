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
@_onnx_symbolic('aten::pixel_unshuffle')
@symbolic_helper.parse_args('v', 'i')
@_beartype.beartype
def pixel_unshuffle(g: jit_utils.GraphContext, self, downscale_factor):
    dims = symbolic_helper._get_tensor_sizes(self)
    if len(dims) != 4:
        return symbolic_helper._unimplemented('pixel_shuffle', 'only support 4d input', self)
    if any((i is None for i in dims[1:])):
        reshape_h = symbolic_helper._reshape_helper(g, symbolic_helper._unsqueeze_helper(g, self, [3]), g.op('Constant', value_t=torch.tensor([0, 0, -1, downscale_factor, 0])), allowzero=0)
        reshape_w = symbolic_helper._reshape_helper(g, reshape_h, g.op('Constant', value_t=torch.tensor([0, 0, 0, 0, -1, downscale_factor])), allowzero=0)
        after_transpose = g.op('Transpose', reshape_w, perm_i=[0, 1, 3, 5, 2, 4])
        final_reshape = symbolic_helper._reshape_helper(g, after_transpose, g.op('Constant', value_t=torch.tensor([0, -1, 1, 1, 0, 0])), allowzero=0)
        return symbolic_helper._squeeze_helper(g, final_reshape, [2, 3])
    else:
        output_channel = dims[1] * downscale_factor * downscale_factor
        after_view = symbolic_helper._reshape_helper(g, self, g.op('Constant', value_t=torch.tensor([-1, dims[1], dims[2] // downscale_factor, downscale_factor, dims[3] // downscale_factor, downscale_factor])), allowzero=0)
        after_transpose = g.op('Transpose', after_view, perm_i=[0, 1, 3, 5, 2, 4])
        return symbolic_helper._reshape_helper(g, after_transpose, g.op('Constant', value_t=torch.tensor([-1, output_channel, dims[2] // downscale_factor, dims[3] // downscale_factor])), allowzero=0)