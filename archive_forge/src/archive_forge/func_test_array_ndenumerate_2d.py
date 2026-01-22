import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_ndenumerate_2d(self):
    arr = np.arange(12).reshape(4, 3)
    arrty = typeof(arr)
    self.assertEqual(arrty.ndim, 2)
    self.assertEqual(arrty.layout, 'C')
    self.assertTrue(arr.flags.c_contiguous)
    self.check_array_ndenumerate_sum(arr, arrty)
    arr = arr.transpose()
    self.assertFalse(arr.flags.c_contiguous)
    self.assertTrue(arr.flags.f_contiguous)
    arrty = typeof(arr)
    self.assertEqual(arrty.layout, 'F')
    self.check_array_ndenumerate_sum(arr, arrty)
    arr = arr[::2]
    self.assertFalse(arr.flags.c_contiguous)
    self.assertFalse(arr.flags.f_contiguous)
    arrty = typeof(arr)
    self.assertEqual(arrty.layout, 'A')
    self.check_array_ndenumerate_sum(arr, arrty)
    arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
    self.check_array_ndenumerate_sum(arr, typeof(arr))