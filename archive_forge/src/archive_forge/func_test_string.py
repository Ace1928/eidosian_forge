import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_string(self):
    """ Storage of fixed-length strings """
    dtypes = tuple((np.dtype(x) for x in ('|S1', '|S10')))
    for dt in dtypes:
        data = np.ndarray((1,), dtype=dt)
        data[...] = 'h'
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertEqual(out.dtype, dt)
        self.assertEqual(out[0], data[0])