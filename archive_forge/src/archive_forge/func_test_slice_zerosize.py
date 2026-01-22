import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_slice_zerosize(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[4:4])