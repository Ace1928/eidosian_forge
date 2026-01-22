from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@overload(target_overloaded_calls_target_overloaded, target='cuda')
def ol_generic_calls_target_overloaded_cuda(x):

    def impl(x):
        x[0] *= CUDA_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)
    return impl