import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_extent_2d(self):
    nparr = np.empty((4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    s, e = arr.extent
    self.assertEqual(e - s, nparr.size * nparr.dtype.itemsize)