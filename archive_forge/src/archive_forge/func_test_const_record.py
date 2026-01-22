import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_const_record(self):
    A = np.zeros(2, dtype=float)
    B = np.zeros(2, dtype=int)
    jcuconst = cuda.jit(cuconstRec).specialize(A, B)
    jcuconst[2, 1](A, B)
    np.testing.assert_allclose(A, CONST_RECORD['x'])
    np.testing.assert_allclose(B, CONST_RECORD['y'])