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
def test_gufunc_auto_transfer(self):
    gufunc = _get_matmulcore_gufunc()
    matrix_ct = 2
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
    dB = cuda.to_device(B)
    C = gufunc(A, dB).copy_to_host()
    Gold = np.matmul(A, B)
    self.assertTrue(np.allclose(C, Gold))