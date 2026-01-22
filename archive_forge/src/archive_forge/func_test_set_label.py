import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_set_label(self):
    self.f['data'].dims[0].label = 'foo'
    self.assertEqual(self.f['data'].dims[2].label, 'x')
    self.assertEqual(self.f['data'].dims[1].label, '')
    self.assertEqual(self.f['data'].dims[0].label, 'foo')