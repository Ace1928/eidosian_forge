from __future__ import annotations
import functools
from typing import Optional
import torch
from torch.onnx import _constants, _type_utils, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('quantized::hardswish')
@_beartype.beartype
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    output = hardswish(g, x)
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)