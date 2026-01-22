import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def pack_float32_to_4bit(array: Union[np.ndarray, Sequence], signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two concecutive elements in a byte.
    See :ref:`onnx-detail-int4` for technical details.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(farray.size/2)` (single dimension).
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)
    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0]))
    single_func = lambda x, y: subbyte.float32x2_to_4bitx2(x, y, signed)
    func = np.frompyfunc(single_func, 2, 1)
    arr = func(array_flat[0::2], array_flat[1::2])
    return arr.astype(np.uint8)