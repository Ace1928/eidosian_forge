import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
def test_ndarray(self):
    """ Dataset auto-creation by direct assignment """
    data = np.ones((4, 4), dtype='f')
    self.f['a'] = data
    self.assertIsInstance(self.f['a'], Dataset)
    self.assertArrayEqual(self.f['a'][...], data)