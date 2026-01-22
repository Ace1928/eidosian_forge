from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_index_2d(self):
    arr = np.arange(3 * 4).reshape(3, 4)
    darr = cuda.to_device(arr)
    x, y = arr.shape
    for i in range(-x, x):
        for j in range(-y, y):
            self.assertEqual(arr[i, j], darr[i, j])
    with self.assertRaises(IndexError):
        darr[-x - 1, 0]
    with self.assertRaises(IndexError):
        darr[x, 0]
    with self.assertRaises(IndexError):
        darr[0, -y - 1]
    with self.assertRaises(IndexError):
        darr[0, y]