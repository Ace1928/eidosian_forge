import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_multiple_output_dtypes(self):

    @guvectorize([void(int32[:], int32[:], float64[:])], '(x)->(x),(x)', target='cuda')
    def copy_and_multiply(A, B, C):
        for i in range(B.size):
            B[i] = A[i]
            C[i] = A[i] * 1.5
    A = np.arange(10, dtype=np.int32) + 1
    B = np.zeros_like(A)
    C = np.zeros_like(A, dtype=np.float64)
    copy_and_multiply(A, B, C)
    np.testing.assert_allclose(A, B)
    np.testing.assert_allclose(A * np.float64(1.5), C)