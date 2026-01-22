import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_block_overruns_extent_error(self):
    mbslice = MultiBlockSlice(start=2, count=2, stride=5, block=4)
    with self.assertRaises(ValueError):
        mbslice.indices(10)