import unittest
import pickle
import numpy as np
from numba import void, float32, float64, int32, int64, jit, guvectorize
from numba.np.ufunc import GUVectorize
from numba.tests.support import tag, TestCase
def test_scalar_input(self):

    @guvectorize(['int32[:], int32[:], int32[:]'], '(n),()->(n)', target=self.target, nopython=True)
    def foo(inp, n, out):
        for i in range(inp.shape[0]):
            out[i] = inp[i] * n[0]
    inp = np.arange(3 * 10, dtype=np.int32).reshape(10, 3)
    out = foo(inp, 2)
    self.assertPreciseEqual(inp * 2, out)