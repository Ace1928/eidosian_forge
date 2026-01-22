import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_tuple_of_array_scalar_tuple(self):

    @cuda.jit
    def f(r, x):
        r[0] = x[0][0]
        r[1] = x[0][1]
        r[2] = x[1]
        r[3] = x[2][0]
        r[4] = x[2][1]
    z = np.arange(2, dtype=np.int64)
    x = (2 * z, 10, (4, 3))
    r = np.zeros(5, dtype=np.int64)
    f[1, 1](r, x)
    self.assertEqual(r[0], 0)
    self.assertEqual(r[1], 2)
    self.assertEqual(r[2], 10)
    self.assertEqual(r[3], 4)
    self.assertEqual(r[4], 3)