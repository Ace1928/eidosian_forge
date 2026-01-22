import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_is_scale(self):
    """Test Dataset.is_scale property"""
    self.assertTrue(self.f['x1'].is_scale)
    self.assertTrue(self.f['x2'].is_scale)
    self.assertTrue(self.f['y1'].is_scale)
    self.assertFalse(self.f['z1'].is_scale)
    self.assertFalse(self.f['data'].is_scale)
    self.assertFalse(self.f['data2'].is_scale)