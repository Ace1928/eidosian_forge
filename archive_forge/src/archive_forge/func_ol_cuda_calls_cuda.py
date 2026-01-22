from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@overload(cuda_calls_cuda, target='cuda')
def ol_cuda_calls_cuda(x):

    def impl(x):
        x[0] *= CUDA_CALLS_CUDA
        cuda_func_1(x)
    return impl