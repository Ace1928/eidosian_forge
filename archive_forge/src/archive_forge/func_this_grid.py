from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@cuda.jit
def this_grid(A):
    cuda.cg.this_grid()
    A[0] = 1.0