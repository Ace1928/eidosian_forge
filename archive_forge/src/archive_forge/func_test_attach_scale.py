import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_attach_scale(self):
    self.f['x3'] = self.f['x2'][...]
    self.f['data'].dims[2].attach_scale(self.f['x3'])
    self.assertEqual(len(self.f['data'].dims[2]), 3)
    self.assertEqual(self.f['data'].dims[2][2], self.f['x3'])