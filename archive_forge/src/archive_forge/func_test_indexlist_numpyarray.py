import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_indexlist_numpyarray(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[np.array([1, 2, 5])])