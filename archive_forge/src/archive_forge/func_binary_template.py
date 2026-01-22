import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def binary_template(self, func, npfunc, npdtype, nprestype, start, stop):
    nelem = 50
    A = np.linspace(start, stop, nelem).astype(npdtype)
    B = np.empty_like(A).astype(nprestype)
    arytype = numpy_support.from_dtype(npdtype)[::1]
    restype = numpy_support.from_dtype(nprestype)[::1]
    cfunc = cuda.jit((arytype, arytype, restype))(func)
    cfunc[1, nelem](A, A, B)
    np.testing.assert_allclose(npfunc(A, A), B)