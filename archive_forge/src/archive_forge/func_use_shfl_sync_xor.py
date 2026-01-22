import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def use_shfl_sync_xor(ary, xor):
    i = cuda.grid(1)
    val = cuda.shfl_xor_sync(4294967295, i, xor)
    ary[i] = val