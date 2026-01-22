from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def is_float16_dtype(dt):
    if dt is None:
        return False
    dt = np.dtype(dt)
    return dt.kind == 'f' and dt.itemsize == 2