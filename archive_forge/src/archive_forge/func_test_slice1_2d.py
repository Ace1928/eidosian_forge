import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_slice1_2d(self):
    nparr = np.empty((4, 5))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    xx = (-2, 0, 2)
    for x in xx:
        expect = nparr[:x]
        got = arr[:x]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)
        self.assertSameContig(got, expect)
    for x, y in itertools.product(xx, xx):
        expect = nparr[:x, :y]
        got = arr[:x, :y]
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)
        self.assertSameContig(got, expect)