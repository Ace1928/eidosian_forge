import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def test_powi(self):
    dec = cuda.jit(void(float64[:, :], int8, float64[:, :]))
    kernel = dec(cu_mat_power)
    power = 2
    A = np.arange(10, dtype=np.float64).reshape(2, 5)
    Aout = np.empty_like(A)
    kernel[1, A.shape](A, power, Aout)
    self.assertTrue(np.allclose(Aout, A ** power))