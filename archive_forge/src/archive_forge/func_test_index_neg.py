import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_index_neg(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[-4])