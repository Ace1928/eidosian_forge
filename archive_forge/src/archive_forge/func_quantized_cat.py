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
@_onnx_symbolic('quantized::cat')
@symbolic_helper.parse_args('v', 'i', 'v', 'v')
@_beartype.beartype
def quantized_cat(g: jit_utils.GraphContext, q_inputs: _C.Value, dim: int, op_scale: _C.Value, op_zero_point: _C.Value) -> _C.Value:
    unpacked_inputs = symbolic_helper._unpack_list(q_inputs)
    dequantized = [symbolic_helper.dequantize_helper(g, input)[0] for input in unpacked_inputs]
    concatenated = g.op('Concat', *dequantized, axis_i=dim)
    return symbolic_helper.quantize_helper(g, concatenated, op_scale, op_zero_point)