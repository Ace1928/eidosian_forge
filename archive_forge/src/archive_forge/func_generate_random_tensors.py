from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def generate_random_tensors(shape: torch.Size, dtype: torch.dtype):
    shape = _convert_symint_to_int_in_shape(shape)
    if dtype == torch.uint8:
        return torch.randint(low=_constants.UINT8_MIN, high=_constants.UINT8_MAX, size=shape, dtype=dtype)
    if dtype == torch.int8:
        return torch.randint(low=_constants.INT8_MIN, high=_constants.INT8_MAX, size=shape, dtype=dtype)
    if dtype == torch.int16:
        return torch.randint(low=_constants.INT16_MIN, high=_constants.INT16_MAX, size=shape, dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(low=_constants.INT32_MIN, high=_constants.INT32_MAX, size=shape, dtype=dtype)
    if dtype == torch.int64:
        return torch.randint(low=_constants.INT64_MIN, high=_constants.INT64_MAX, size=shape, dtype=dtype)
    if dtype == torch.bool:
        random_numbers = torch.rand(shape)
        return torch.where(random_numbers > 0.5, torch.tensor(True), torch.tensor(False))
    if fx_type_utils.is_torch_complex_dtype(dtype):
        return torch.view_as_complex(torch.randn((*shape, 2), dtype=fx_type_utils.from_complex_to_float(dtype)))
    return torch.randn(shape, dtype=dtype)