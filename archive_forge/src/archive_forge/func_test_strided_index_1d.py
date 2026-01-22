from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_strided_index_1d(self):
    arr = np.arange(10)
    darr = cuda.to_device(arr)
    for i in range(arr.size):
        np.testing.assert_equal(arr[i::2], darr[i::2].copy_to_host())