import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
@cuda.jit('void(double[:], double[:])')
def copykernel(x, y):
    i = cuda.grid(1)
    if i < x.shape[0]:
        x[i] = i
        y[i] = i