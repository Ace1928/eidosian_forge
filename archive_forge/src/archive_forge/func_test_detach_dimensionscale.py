import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_detach_dimensionscale(self):
    self.assertTrue(h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2))
    h5py.h5ds.detach_scale(self.f['data'].id, self.f['x1'].id, 2)
    self.assertFalse(h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2))
    self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 2), 1)