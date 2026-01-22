import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_reduce_async(self):

    @vectorize(signatures, target='cuda')
    def vector_add(a, b):
        return a + b
    stream = cuda.stream()
    dtype = np.int32
    for n in input_sizes:
        x = np.arange(n, dtype=dtype)
        expected = np.add.reduce(x)
        dx = cuda.to_device(x, stream)
        actual = vector_add.reduce(dx, stream=stream)
        np.testing.assert_allclose(expected, actual)
        self.assertEqual(dtype, actual.dtype)