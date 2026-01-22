import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_unituple(self):

    @cuda.jit
    def f(r, x):
        r[0] = x[0]
        r[1] = x[1]
        r[2] = x[2]
    x = (1, 2, 3)
    r = np.zeros(len(x), dtype=np.int64)
    f[1, 1](r, x)
    for i in range(len(x)):
        self.assertEqual(r[i], x[i])