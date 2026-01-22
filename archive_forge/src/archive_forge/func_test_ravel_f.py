import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_ravel_f(self):
    ary = np.arange(60)
    reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
    for order in 'FA':
        expect = reshaped.ravel(order=order)
        dary = cuda.to_device(reshaped)
        dflat = dary.ravel(order=order)
        flat = dflat.copy_to_host()
        self.assertTrue(dary is not dflat)
        self.assertEqual(flat.ndim, 1)
        self.assertPreciseEqual(expect, flat)