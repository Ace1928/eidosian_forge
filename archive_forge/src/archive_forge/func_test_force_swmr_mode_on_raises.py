import numpy as np
import h5py
from .common import ut, TestCase
def test_force_swmr_mode_on_raises(self):
    """ Verify when reading a file cannot be forcibly switched to swmr mode.
        When reading with SWMR the file must be opened with swmr=True."""
    with self.assertRaises(Exception):
        self.f.swmr_mode = True
    self.assertTrue(self.f.swmr_mode)