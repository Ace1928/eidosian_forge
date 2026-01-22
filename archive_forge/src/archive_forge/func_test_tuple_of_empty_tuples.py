import numpy as np
from collections import namedtuple
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_tuple_of_empty_tuples(self):

    @cuda.jit
    def f(r, x):
        r[0] = len(x)
        r[1] = len(x[0])
    x = ((), (), ())
    r = np.ones(2, dtype=np.int64)
    f[1, 1](r, x)
    self.assertEqual(r[0], 3)
    self.assertEqual(r[1], 0)