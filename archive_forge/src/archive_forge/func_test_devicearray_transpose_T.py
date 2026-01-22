import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_transpose_T(self):
    original = np.array(np.arange(12)).reshape(3, 4)
    array = cuda.to_device(original).T.copy_to_host()
    self.assertTrue(np.all(array == original.T))