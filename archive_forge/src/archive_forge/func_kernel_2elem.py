import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def kernel_2elem(res):
    v = vobj(base_type(0), base_type(1))
    res[0] = v.x
    res[1] = v.y