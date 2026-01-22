import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('DeviceNDArray class not present in simulator')
def test_devicearray_relaxed_strides(self):
    arr = devicearray.DeviceNDArray((1, 10), (800, 8), np.float64)
    self.assertTrue(arr.flags['C_CONTIGUOUS'])
    self.assertTrue(arr.flags['F_CONTIGUOUS'])