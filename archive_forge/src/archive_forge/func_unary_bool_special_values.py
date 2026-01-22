import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def unary_bool_special_values(self, func, npfunc, npdtype, npmtype):
    fi = np.finfo(npdtype)
    denorm = fi.tiny / 4
    A = np.array([0.0, denorm, fi.tiny, 0.5, 1.0, fi.max, np.inf, np.nan], dtype=npdtype)
    B = np.empty_like(A, dtype=np.int32)
    cfunc = cuda.jit((npmtype[::1], int32[::1]))(func)
    cfunc[1, A.size](A, B)
    np.testing.assert_array_equal(B, npfunc(A))
    cfunc[1, A.size](-A, B)
    np.testing.assert_array_equal(B, npfunc(-A))