import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def kernel_1elem(res):
    v = vobj(base_type(0))
    res[0] = v.x