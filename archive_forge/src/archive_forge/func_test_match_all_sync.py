import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
@unittest.skipUnless(_safe_cc_check((7, 0)), 'Matching requires at least Volta Architecture')
def test_match_all_sync(self):
    compiled = cuda.jit('void(int32[:], int32[:])')(use_match_all_sync)
    nelem = 10
    ary_in = np.zeros(nelem, dtype=np.int32)
    ary_out = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 1023))
    ary_in[1] = 4
    compiled[1, nelem](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 0))