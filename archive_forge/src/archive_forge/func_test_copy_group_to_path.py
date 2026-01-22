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
def test_copy_group_to_path(self):
    foo = self.f1.create_group('foo')
    foo['bar'] = [1, 2, 3]
    self.f1.copy(foo, 'baz')
    baz = self.f1['baz']
    self.assertIsInstance(baz, Group)
    self.assertArrayEqual(baz['bar'], np.array([1, 2, 3]))
    self.f2.copy(foo, 'foo')
    self.assertIsInstance(self.f2['/foo'], Group)
    self.assertArrayEqual(self.f2['foo/bar'], np.array([1, 2, 3]))