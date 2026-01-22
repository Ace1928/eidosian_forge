import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
def test_no_zero_division_error(self):

    @cuda.jit
    def f(r, x, y):
        r[0] = y[0] / x[0]
        r[1] = y[0]
    r = np.zeros(2)
    x = np.zeros(1)
    y = np.ones(1)
    f[1, 1](r, x, y)
    self.assertTrue(np.isinf(r[0]), 'Expected inf from div by zero')
    self.assertEqual(r[1], y[0], 'Expected execution to continue')