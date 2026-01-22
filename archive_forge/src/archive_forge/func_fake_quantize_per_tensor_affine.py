from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::fake_quantize_per_tensor_affine')
@symbolic_helper.parse_args('v', 'v', 'v', 'i', 'i')
@_beartype.beartype
def fake_quantize_per_tensor_affine(g: jit_utils.GraphContext, inputs, scale, zero_point, quant_min=-128, quant_max=127):
    if (quant_min, quant_max) == (0, 127):
        symbolic_helper._onnx_opset_unsupported_detailed('fake_quantize_per_tensor_affine', 10, 13, 'Quantize range (0, 127) not supported, requires opset 13 Clip', inputs)
    if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
        raise errors.SymbolicValueError(f'For (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). Got ({quant_min}, {quant_max})', inputs)
    scale = symbolic_helper._maybe_get_scalar(scale)
    if scale is None:
        symbolic_helper._onnx_opset_unsupported_detailed('fake_quantize_per_tensor_affine', 10, 13, 'Non-constant scale not supported', inputs)
    scale = scale.float().data
    if quant_min == 0:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    return g.op('DequantizeLinear', g.op('QuantizeLinear', inputs, scale, zero_point), scale, zero_point)