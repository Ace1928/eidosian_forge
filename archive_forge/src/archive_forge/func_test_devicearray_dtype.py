import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_dtype(self):
    dary = cuda.device_array(shape=(100,), dtype='f4')
    self.assertEqual(dary.dtype, np.dtype('f4'))