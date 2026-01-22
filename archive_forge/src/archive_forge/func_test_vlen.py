import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def test_vlen(self):
    a = np.array([np.arange(3), np.arange(4)], dtype=h5t.vlen_dtype(int))
    self.f.attrs['a'] = a
    self.assertArrayEqual(self.f.attrs['a'][0], a[0])