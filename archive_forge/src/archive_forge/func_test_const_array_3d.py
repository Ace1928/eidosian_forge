import numpy as np
from numba import cuda, complex64, int32, float64
from numba.cuda.testing import unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
def test_const_array_3d(self):
    sig = (complex64[:, :, :],)
    jcuconst3d = cuda.jit(sig)(cuconst3d)
    A = np.zeros_like(CONST3D, order='F')
    jcuconst3d[1, (5, 5, 5)](A)
    self.assertTrue(np.all(A == CONST3D))
    if not ENABLE_CUDASIM:
        asm = jcuconst3d.inspect_asm(sig)
        complex_load = 'ld.const.v2.f32'
        description = 'Load the complex as a vector of 2x f32'
        self.assertIn(complex_load, asm, description)