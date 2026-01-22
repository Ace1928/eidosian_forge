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
def test_get_class(self):
    """ Object class is returned with getclass option """
    self.f.create_group('foo')
    out = self.f.get('foo', getclass=True)
    self.assertEqual(out, Group)
    self.f.create_dataset('bar', (4,))
    out = self.f.get('bar', getclass=True)
    self.assertEqual(out, Dataset)
    self.f['baz'] = np.dtype('|S10')
    out = self.f.get('baz', getclass=True)
    self.assertEqual(out, Datatype)