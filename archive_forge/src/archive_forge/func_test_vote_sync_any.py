import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def test_vote_sync_any(self):
    compiled = cuda.jit('void(int32[:], int32[:])')(use_vote_sync_any)
    nelem = 32
    ary_in = np.zeros(nelem, dtype=np.int32)
    ary_out = np.empty(nelem, dtype=np.int32)
    compiled[1, nelem](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 0))
    ary_in[2] = 1
    ary_in[5] = 1
    compiled[1, nelem](ary_in, ary_out)
    self.assertTrue(np.all(ary_out == 1))