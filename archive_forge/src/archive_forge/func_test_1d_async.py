import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_1d_async(self):

    @vectorize(signatures, target='cuda')
    def vector_add(a, b):
        return a + b
    stream = cuda.stream()
    for ty in dtypes:
        data = np.array(np.random.random(self.N), dtype=ty)
        device_data = cuda.to_device(data, stream)
        dresult = vector_add(device_data, device_data, stream=stream)
        actual = dresult.copy_to_host()
        expected = np.add(data, data)
        np.testing.assert_allclose(expected, actual)
        self.assertEqual(actual.dtype, ty)