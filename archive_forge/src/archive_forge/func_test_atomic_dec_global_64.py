import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_dec_global_64(self):
    rand_const, ary, idx = self.inc_dec_1dim_setup(dtype=np.uint64)
    sig = 'void(uint64[:], uint64[:], uint64)'
    self.check_dec_index2(ary, idx, rand_const, sig, 1, 32, atomic_dec_global)