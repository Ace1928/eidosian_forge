import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def encode3d(arr, marker=-2147483648, bit_depth=32, size_max=1024):
    if arr.ndim != 3:
        raise ValueError('only 3d arr suppported')
    if bit_depth not in [32, 64]:
        raise ValueError('only bit_depth of 32 or 64 is supported')
    if size_max > 1024:
        dtype = np.int64
    else:
        dtype = np.int32
    image = cupy.zeros(arr.shape, dtype=dtype, order='C')
    kern = _get_encode3d_kernel(size_max, marker=marker)
    kern(arr, image, size=image.size)
    return image