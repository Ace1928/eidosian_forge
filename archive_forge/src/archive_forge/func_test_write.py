import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_write(self):
    """ ValueError on string write wipes out attribute """
    s = b'Hello\x00Hello'
    try:
        self.f.attrs['x'] = s
    except ValueError:
        pass
    with self.assertRaises(KeyError):
        self.f.attrs['x']