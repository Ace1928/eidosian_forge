from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_index_1d(self):
    arr = np.arange(10)
    darr = cuda.to_device(arr)
    x, = arr.shape
    for i in range(-x, x):
        self.assertEqual(arr[i], darr[i])
    with self.assertRaises(IndexError):
        darr[-x - 1]
    with self.assertRaises(IndexError):
        darr[x]