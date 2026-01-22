import functools
import torch
from torch.nn.functional import (
from torch.onnx import _type_utils, errors, symbolic_helper, utils
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::grid_sampler')
@symbolic_helper.parse_args('v', 'v', 'i', 'i', 'b')
@_beartype.beartype
def grid_sampler(g: jit_utils.GraphContext, input, grid, mode_enum, padding_mode_enum, align_corners):
    if symbolic_helper._get_tensor_rank(input) == 5:
        return symbolic_helper._onnx_unsupported('GridSample with 5D volumetric input')
    mode_s = {v: k for k, v in GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]
    padding_mode_s = {v: k for k, v in GRID_SAMPLE_PADDING_MODES.items()}[padding_mode_enum]
    return g.op('GridSample', input, grid, align_corners_i=int(align_corners), mode_s=mode_s, padding_mode_s=padding_mode_s)