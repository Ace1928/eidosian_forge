import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_const_array_2d(self):
    sig = (int32[:, :],)
    jcuconst2d = cuda.jit(sig)(cuconst2d)
    A = np.zeros_like(CONST2D, order='C')
    jcuconst2d[(2, 2), (5, 5)](A)
    self.assertTrue(np.all(A == CONST2D))
    if not ENABLE_CUDASIM:
        self.assertIn('ld.const.u32', jcuconst2d.inspect_asm(sig), 'load the ints as ints')