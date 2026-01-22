import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def math_pow_binop(A, B, C):
    i = cuda.grid(1)
    C[i] = A[i] ** B[i]