from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_prefix_2d(self):
    arr = np.arange(3 ** 2).reshape(3, 3)
    darr = cuda.to_device(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            expect = arr[i:, j:]
            sliced = darr[i:, j:]
            self.assertEqual(expect.shape, sliced.shape)
            self.assertEqual(expect.strides, sliced.strides)
            got = sliced.copy_to_host()
            self.assertTrue(np.all(expect == got))