import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def float8e5m2_to_float32(ival: int) -> Any:
    if ival < 0 or ival > 255:
        raise ValueError(f'{ival} is not a float8.')
    if ival in (255, 254, 253):
        return np.float32(-np.nan)
    if ival in (127, 126, 125):
        return np.float32(np.nan)
    if ival == 252:
        return -np.float32(np.inf)
    if ival == 124:
        return np.float32(np.inf)
    if ival & 127 == 0:
        return np.float32(0)
    sign = ival & 128
    ival &= 127
    expo = ival >> 2
    mant = ival & 3
    powe = expo & 31
    if expo == 0:
        powe -= 14
        fraction = 0
    else:
        powe -= 15
        fraction = 1
    fval = float(mant / 4 + fraction) * 2.0 ** powe
    if sign:
        fval = -fval
    return np.float32(fval)