import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_reshape_2d1d(self):
    nparr = np.empty((4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    expect = nparr.reshape(5 * 4)
    got = arr.reshape(5 * 4)[0]
    self.assertEqual(got.shape, expect.shape)
    self.assertEqual(got.strides, expect.strides)