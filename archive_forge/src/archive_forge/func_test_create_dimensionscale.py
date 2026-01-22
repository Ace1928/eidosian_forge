import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_create_dimensionscale(self):
    """ Create a dimension scale from existing dataset """
    self.assertTrue(h5py.h5ds.is_scale(self.f['x1'].id))
    self.assertEqual(h5py.h5ds.get_scale_name(self.f['x1'].id), b'')
    self.assertEqual(self.f['x1'].attrs['CLASS'], b'DIMENSION_SCALE')
    self.assertEqual(h5py.h5ds.get_scale_name(self.f['x2'].id), b'x2 name')