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
def test_intermediate_create_dataset(self):
    """ Intermediate is created if it doesn't exist """
    dt = h5py.string_dtype()
    self.f.require_dataset('foo/bar/baz', (1,), dtype=dt)
    group = self.f.get('foo')
    assert isinstance(group, Group)
    group = self.f.get('foo/bar')
    assert isinstance(group, Group)