from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
def test_target_overloaded_calls_target_overloaded(self):

    def kernel(x):
        target_overloaded_calls_target_overloaded(x)
    expected = CUDA_TARGET_OL_CALLS_TARGET_OL * CUDA_TARGET_OL
    self.check_overload(kernel, expected)
    expected = GENERIC_TARGET_OL_CALLS_TARGET_OL * GENERIC_TARGET_OL
    self.check_overload_cpu(kernel, expected)