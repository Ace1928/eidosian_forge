from h5py import File
from h5py._hl.base import is_hdf5, Empty
from .common import ut, TestCase, UNICODE_FILENAMES
import numpy as np
import os
import tempfile
def test_namedtype(self):
    """ Named type repr() with unicode """
    self.f['type'] = np.dtype('f')
    typ = self.f['type']
    self._check_type(typ)