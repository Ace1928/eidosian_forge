from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_call_target_overloaded(self):

    def kernel(x):
        target_overloaded(x)
    expected = CUDA_TARGET_OL
    self.check_overload(kernel, expected)