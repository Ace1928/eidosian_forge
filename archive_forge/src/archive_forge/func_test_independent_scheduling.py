import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
@unittest.skipUnless(_safe_cc_check((7, 0)), 'Independent scheduling requires at least Volta Architecture')
def test_independent_scheduling(self):
    compiled = cuda.jit('void(uint32[:])')(use_independent_scheduling)
    arr = np.empty(32, dtype=np.uint32)
    exp = np.tile((286331153, 572662306, 1145324612, 2290649224), 8)
    compiled[1, 32](arr)
    self.assertTrue(np.all(arr == exp))