import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def modf_template_nan(dtype, arytype):
    A = np.array([np.nan], dtype=dtype)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
    cfunc[1, len(A)](A, B, C)
    self.assertTrue(np.isnan(B))
    self.assertTrue(np.isnan(C))