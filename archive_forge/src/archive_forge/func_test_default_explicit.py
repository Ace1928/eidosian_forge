import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_default_explicit(self):
    mbslice = MultiBlockSlice(start=0, count=10, stride=1, block=1)
    self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
    np.testing.assert_array_equal(self.dset[mbslice], self.arr)