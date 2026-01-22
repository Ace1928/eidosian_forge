import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_scalar_null(self):
    """ Slicing with [()] yields array scalar """
    dset = self.f.create_dataset('x', shape=(), dtype='i1')
    out = dset[()]
    self.assertIsInstance(out, np.int8)