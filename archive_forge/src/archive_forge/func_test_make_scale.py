import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_make_scale(self):
    self.f['x1'].make_scale(b'foobar')
    self.assertEqual(self.f['data'].dims[2]['foobar'], self.f['x1'])
    self.f['data2'].make_scale(b'foobaz')
    self.f['data'].dims[2].attach_scale(self.f['data2'])
    self.assertEqual(self.f['data'].dims[2]['foobaz'], self.f['data2'])