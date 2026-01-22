from numba import cuda
import numpy as np
from numba.cuda.testing import CUDATestCase
from numba.tests.support import override_config
import unittest
def test_device_array(self):

    @cuda.jit
    def foo(x, y):
        i = cuda.grid(1)
        y[i] = x[i]
    x = np.arange(10)
    y = np.empty_like(x)
    dx = cuda.to_device(x)
    dy = cuda.to_device(y)
    foo[10, 1](dx, dy)
    dy.copy_to_host(y)
    self.assertTrue(np.all(x == y))