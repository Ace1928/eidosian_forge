from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
@unittest.skip('Needs insert_unresolved_ref support in target')
def test_type_change(self):
    pfunc = self.mod.type_change_self.py_func
    cfunc = self.mod.type_change_self

    @cuda.jit
    def kernel(r, x, y):
        r[0] = cfunc(x[0], y[0])
    args = (13, 0.125)
    x = np.asarray([args[0]], dtype=np.int64)
    y = np.asarray([args[1]], dtype=np.float64)
    r = np.zeros_like(x)
    kernel[1, 1](r, x, y)
    expected = pfunc(*args)
    actual = r[0]
    self.assertPreciseEqual(actual, expected)