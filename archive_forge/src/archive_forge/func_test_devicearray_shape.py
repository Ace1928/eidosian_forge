import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_shape(self):
    ary = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    dary = cuda.to_device(ary)
    self.assertEqual(ary.shape, dary.shape)
    self.assertEqual(ary.shape[1:], dary.shape[1:])