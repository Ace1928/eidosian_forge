import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def dyn_shared_memory(ary):
    i = cuda.grid(1)
    sm = cuda.shared.array(0, float32)
    sm[i] = i * 2
    cuda.syncthreads()
    ary[i] = sm[i]