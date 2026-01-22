import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_tuple_of_arrays(self):

    @cuda.jit
    def f(x):
        i = cuda.grid(1)
        if i < len(x[0]):
            x[0][i] = x[1][i] + x[2][i]
    N = 10
    x0 = np.zeros(N)
    x1 = np.ones_like(x0)
    x2 = x1 * 3
    x = (x0, x1, x2)
    f[1, N](x)
    np.testing.assert_equal(x0, x1 + x2)