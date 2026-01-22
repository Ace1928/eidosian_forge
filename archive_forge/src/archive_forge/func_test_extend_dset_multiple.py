import numpy as np
import h5py
from .common import ut, TestCase
def test_extend_dset_multiple(self):
    self.f.swmr_mode = True
    self.assertTrue(self.f.swmr_mode)
    self.dset.resize((4,))
    self.dset[0:] = self.data
    self.dset.flush()
    self.dset.refresh()
    self.assertArrayEqual(self.dset, self.data)
    self.dset.resize((8,))
    self.dset[4:] = self.data
    self.dset.flush()
    self.dset.refresh()
    self.assertArrayEqual(self.dset[0:4], self.data)
    self.assertArrayEqual(self.dset[4:8], self.data)