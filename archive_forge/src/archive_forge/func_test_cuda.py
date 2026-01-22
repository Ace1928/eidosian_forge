from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_cuda(self):

    def kernel(x):
        cuda_func_1(x)
    expected = CUDA_FUNCTION_1
    self.check_overload(kernel, expected)