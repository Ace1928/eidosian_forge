import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_stride_more_than_block(self):
    mbslice = MultiBlockSlice(stride=3, block=2)
    self.assertEqual(mbslice.indices(10), (0, 3, 3, 2))
    np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 1, 3, 4, 6, 7]))