import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_squeeze_invalid_axis(self):
    nparr = np.empty((1, 2, 1, 4, 1, 3))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    with self.assertRaises(ValueError):
        arr.squeeze(axis=1)
    with self.assertRaises(ValueError):
        arr.squeeze(axis=(2, 3))