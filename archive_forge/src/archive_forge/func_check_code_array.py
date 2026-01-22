from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
import numpy as np
from contourpy import FillType, LineType
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.types import MOVETO, code_dtype, offset_dtype, point_dtype
def check_code_array(codes: Any) -> None:
    if not isinstance(codes, np.ndarray):
        raise TypeError(f'Expected numpy array not {type(codes)}')
    if codes.dtype != code_dtype:
        raise ValueError(f'Expected numpy array of dtype {code_dtype} not {codes.dtype}')
    if not (codes.ndim == 1 and len(codes) > 1):
        raise ValueError(f'Expected numpy array of shape (?,) not {codes.shape}')
    if codes[0] != MOVETO:
        raise ValueError(f'First element of code array must be {MOVETO}, not {codes[0]}')