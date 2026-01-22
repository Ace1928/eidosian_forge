from numba import cuda, njit
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
import numpy as np
@overload(generic_func_1, target='generic')
def ol_generic_func_1(x):

    def impl(x):
        x[0] *= GENERIC_FUNCTION_1
    return impl