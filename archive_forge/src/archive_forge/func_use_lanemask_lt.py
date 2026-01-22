import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
@cuda.jit
def use_lanemask_lt(x):
    i = cuda.grid(1)
    x[i] = cuda.lanemask_lt()