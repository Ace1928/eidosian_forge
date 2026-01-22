import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
@skip_on_cudasim('syncwarp not implemented on cudasim')
@unittest.skipUnless(_safe_cc_check((7, 0)), 'Partial masks require CC 7.0 or greater')
def test_coop_syncwarp(self):
    expected = 496
    nthreads = 32
    nblocks = 1
    compiled = cuda.jit('void(int32[::1])')(coop_syncwarp)
    res = np.zeros(1, dtype=np.int32)
    compiled[nblocks, nthreads](res)
    np.testing.assert_equal(expected, res[0])