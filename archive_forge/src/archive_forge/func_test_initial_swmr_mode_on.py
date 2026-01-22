import numpy as np
import h5py
from .common import ut, TestCase
def test_initial_swmr_mode_on(self):
    """ Verify that the file is initially in SWMR mode"""
    self.assertTrue(self.f.swmr_mode)