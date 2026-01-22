import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_contiguous_copy_device(self):
    a_c = np.arange(5 * 5 * 5).reshape(5, 5, 5)
    a_f = np.array(a_c, order='F')
    self.assertTrue(a_c.flags.c_contiguous)
    self.assertTrue(a_f.flags.f_contiguous)
    d = cuda.to_device(a_c)
    with self.assertRaises(ValueError) as e:
        d.copy_to_device(cuda.to_device(a_f))
    self.assertEqual('incompatible strides: {} vs. {}'.format(a_c.strides, a_f.strides), str(e.exception))
    d.copy_to_device(cuda.to_device(a_c))
    self.assertTrue(np.all(d.copy_to_host() == a_c))
    d = cuda.to_device(a_f)
    with self.assertRaises(ValueError) as e:
        d.copy_to_device(cuda.to_device(a_c))
    self.assertEqual('incompatible strides: {} vs. {}'.format(a_f.strides, a_c.strides), str(e.exception))
    d.copy_to_device(cuda.to_device(a_f))
    self.assertTrue(np.all(d.copy_to_host() == a_f))