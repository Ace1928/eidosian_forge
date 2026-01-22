import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_replace(self):
    N = 100
    array = np.arange(N, dtype=np.int32)
    original = array.copy()
    gpumem = cuda.to_device(array)
    cuda.to_device(array * 2, to=gpumem)
    gpumem.copy_to_host(array)
    np.testing.assert_array_equal(array, original * 2)