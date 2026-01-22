import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_inc_global_2_32(self):
    rand_const, ary = self.inc_dec_2dim_setup(np.uint32)
    sig = 'void(uint32[:,:], uint32)'
    self.check_inc(ary, rand_const, sig, 1, (4, 8), atomic_inc_global_2)