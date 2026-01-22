import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_squeeze(self):
    nparr = np.empty((1, 2, 1, 4, 1, 3))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)

    def _assert_equal_shape_strides(arr1, arr2):
        self.assertEqual(arr1.shape, arr2.shape)
        self.assertEqual(arr1.strides, arr2.strides)
    _assert_equal_shape_strides(arr, nparr)
    _assert_equal_shape_strides(arr.squeeze()[0], nparr.squeeze())
    for axis in (0, 2, 4, (0, 2), (0, 4), (2, 4), (0, 2, 4)):
        _assert_equal_shape_strides(arr.squeeze(axis=axis)[0], nparr.squeeze(axis=axis))