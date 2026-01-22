import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
@cuda.jit(void(float64[::1], int64, int64))
def test_0_0(r, x, y):
    r[0] = math.remainder(x, y)