import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray(self):
    array = np.arange(100, dtype=np.int32)
    original = array.copy()
    gpumem = cuda.to_device(array)
    array[:] = 0
    gpumem.copy_to_host(array)
    np.testing.assert_array_equal(array, original)