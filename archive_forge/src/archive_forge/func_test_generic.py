from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_generic(self):

    def kernel(x):
        generic_func_1(x)
    expected = GENERIC_FUNCTION_1
    self.check_overload(kernel, expected)