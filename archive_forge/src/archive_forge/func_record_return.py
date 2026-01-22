from numba import cuda
from numba.cuda.testing import CUDATestCase
import numpy as np
import sys
@cuda.jit(cache=True)
def record_return(r, ary, i):
    r[()] = ary[i]