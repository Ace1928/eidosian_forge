from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
def test_empty_slice_2d(self):
    arr = np.arange(5 * 7).reshape(5, 7)
    darr = cuda.to_device(arr)
    np.testing.assert_array_equal(darr[:0].copy_to_host(), arr[:0])
    np.testing.assert_array_equal(darr[3, :0].copy_to_host(), arr[3, :0])
    self.assertFalse(darr[:0][:0].copy_to_host())
    np.testing.assert_array_equal(darr[:0][:1].copy_to_host(), arr[:0][:1])
    np.testing.assert_array_equal(darr[:0][-1:].copy_to_host(), arr[:0][-1:])