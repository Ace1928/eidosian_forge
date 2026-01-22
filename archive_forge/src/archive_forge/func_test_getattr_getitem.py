import numpy as np
from numba import int8, int16, int32
from numba import cuda, vectorize, njit
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.enum_usecases import (
def test_getattr_getitem(self):

    def f(out):
        out[0] = Color.red == Color.green
        out[1] = Color['red'] == Color['green']
    cuda_f = cuda.jit(f)
    got = np.zeros((2,), dtype=np.bool_)
    expected = got.copy()
    cuda_f[1, 1](got)
    f(expected)
    self.assertPreciseEqual(expected, got)