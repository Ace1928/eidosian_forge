import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_partition(self):
    N = 100
    array = np.arange(N, dtype=np.int32)
    original = array.copy()
    gpumem = cuda.to_device(array)
    left, right = gpumem.split(N // 2)
    array[:] = 0
    self.assertTrue(np.all(array == 0))
    right.copy_to_host(array[N // 2:])
    left.copy_to_host(array[:N // 2])
    self.assertTrue(np.all(array == original))