from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
@njit
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)