import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_const_array(self):
    sig = (float64[:],)
    jcuconst = cuda.jit(sig)(cuconst)
    A = np.zeros_like(CONST1D)
    jcuconst[2, 5](A)
    self.assertTrue(np.all(A == CONST1D + 1))
    if not ENABLE_CUDASIM:
        self.assertIn('ld.const.f64', jcuconst.inspect_asm(sig), "as we're adding to it, load as a double")