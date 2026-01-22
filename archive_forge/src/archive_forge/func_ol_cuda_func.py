from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@overload(cuda_func_2, target='cuda')
def ol_cuda_func(x):

    def impl(x):
        x[0] *= CUDA_FUNCTION_2
    return impl