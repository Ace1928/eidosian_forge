import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def use_shfl_sync_down(ary, delta):
    i = cuda.grid(1)
    val = cuda.shfl_down_sync(4294967295, i, delta)
    ary[i] = val