import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_multi(self):
    """ Rank-1 arrays are correctly recovered """
    data = np.ndarray((42,), dtype='f')
    data[:] = 42.0
    data[10:35] = -47.0
    self.f.attrs['x'] = data
    out = self.f.attrs['x']
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, (42,))
    self.assertArrayEqual(out, data)