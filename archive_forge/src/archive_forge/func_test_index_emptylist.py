import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_index_emptylist(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[:, []])
    self.assertNumpyBehavior(self.dset, self.data, np.s_[[]])