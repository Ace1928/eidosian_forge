import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def inc_dec_2dim_setup(self, dtype):
    rconst = np.random.randint(32, dtype=dtype)
    rary = np.random.randint(0, 32, size=32).astype(dtype).reshape(4, 8)
    return (rconst, rary)