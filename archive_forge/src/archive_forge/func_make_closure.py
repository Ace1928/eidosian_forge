from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
def make_closure(x):

    @cuda.jit(cache=True)
    def closure(r, y):
        r[()] = x + y[()]
    return CUDAUseCase(closure)