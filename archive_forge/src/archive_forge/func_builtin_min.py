import numpy as np
from numba import cuda, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def builtin_min(A, B, C):
    i = cuda.grid(1)
    if i >= len(C):
        return
    C[i] = float64(min(A[i], B[i]))