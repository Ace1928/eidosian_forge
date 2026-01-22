import functools
import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::nonzero_numpy')
@_beartype.beartype
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=None):
    return unbind(g, opset9.nonzero(g, input), 1, _outputs=_outputs)