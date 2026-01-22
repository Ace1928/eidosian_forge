from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
def check_fib(self, cfunc):

    @cuda.jit
    def kernel(r, x):
        r[0] = cfunc(x[0])
    x = np.asarray([10], dtype=np.int64)
    r = np.zeros_like(x)
    kernel[1, 1](r, x)
    actual = r[0]
    expected = 55
    self.assertPreciseEqual(actual, expected)