import unittest
import itertools
import numpy as np
from numba.misc.dummyarray import Array
def test_strided_3d(self):
    nparr = np.empty((4, 5, 6))
    arr = Array.from_desc(0, nparr.shape, nparr.strides, nparr.dtype.itemsize)
    xx = (-2, -1, 1, 2)
    for a, b, c in itertools.product(xx, xx, xx):
        expect = nparr[::a, ::b, ::c]
        got = arr[::a, ::b, ::c]
        self.assertSameContig(got, expect)
        self.assertEqual(got.shape, expect.shape)
        self.assertEqual(got.strides, expect.strides)