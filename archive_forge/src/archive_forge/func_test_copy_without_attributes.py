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
def test_copy_without_attributes(self):
    self.f1['foo'] = [1, 2, 3]
    foo = self.f1['foo']
    foo.attrs['bar'] = [4, 5, 6]
    self.f1.copy(foo, 'baz', without_attrs=True)
    self.assertArrayEqual(self.f1['baz'], np.array([1, 2, 3]))
    assert 'bar' not in self.f1['baz'].attrs
    self.f2.copy(foo, 'baz', without_attrs=True)
    self.assertArrayEqual(self.f2['baz'], np.array([1, 2, 3]))
    assert 'bar' not in self.f2['baz'].attrs