import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_empty_tuple(self):

    @cuda.jit
    def f(r, x):
        r[0] = len(x)
    x = tuple()
    r = np.ones(1, dtype=np.int64)
    f[1, 1](r, x)
    self.assertEqual(r[0], 0)