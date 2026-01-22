import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_contigous_4d(self):
    ary = np.arange(60)
    cary = ary.reshape(2, 5, 2, 3)
    fary = np.asfortranarray(cary)
    dcary = cuda.to_device(cary)
    dfary = cuda.to_device(fary)
    self.assertTrue(dcary.is_c_contiguous())
    self.assertTrue(not dfary.is_c_contiguous())
    self.assertTrue(not dcary.is_f_contiguous())
    self.assertTrue(dfary.is_f_contiguous())