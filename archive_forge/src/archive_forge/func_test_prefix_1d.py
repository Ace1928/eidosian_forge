from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_prefix_1d(self):
    arr = np.arange(5)
    darr = cuda.to_device(arr)
    for i in range(arr.size):
        expect = arr[i:]
        got = darr[i:].copy_to_host()
        self.assertTrue(np.all(expect == got))