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
def test_copy_shallow(self):
    foo = self.f1.create_group('foo')
    bar = foo.create_group('bar')
    foo['qux'] = [1, 2, 3]
    bar['quux'] = [4, 5, 6]
    self.f1.copy(foo, 'baz', shallow=True)
    baz = self.f1['baz']
    self.assertIsInstance(baz, Group)
    self.assertIsInstance(baz['bar'], Group)
    self.assertEqual(len(baz['bar']), 0)
    self.assertArrayEqual(baz['qux'], np.array([1, 2, 3]))
    self.f2.copy(foo, 'foo', shallow=True)
    self.assertIsInstance(self.f2['/foo'], Group)
    self.assertIsInstance(self.f2['foo/bar'], Group)
    self.assertEqual(len(self.f2['foo/bar']), 0)
    self.assertArrayEqual(self.f2['foo/qux'], np.array([1, 2, 3]))