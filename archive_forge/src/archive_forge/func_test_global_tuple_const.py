import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import unittest, CUDATestCase
@unittest.SkipTest
def test_global_tuple_const(self):
    """Test coop_smem2d
        """
    compiled = cuda.jit('void(float32[:,:])')(coop_smem2d)
    shape = (10, 20)
    ary = np.empty(shape, dtype=np.float32)
    compiled[1, shape](ary)
    exp = np.empty_like(ary)
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            exp[i, j] = float(i + 1) / (j + 1)
    self.assertTrue(np.allclose(ary, exp))