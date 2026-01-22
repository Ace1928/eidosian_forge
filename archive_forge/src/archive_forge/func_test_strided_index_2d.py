from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_strided_index_2d(self):
    arr = np.arange(6 * 7).reshape(6, 7)
    darr = cuda.to_device(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            np.testing.assert_equal(arr[i::2, j::2], darr[i::2, j::2].copy_to_host())