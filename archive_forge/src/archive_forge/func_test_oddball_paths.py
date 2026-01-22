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
def test_oddball_paths(self):
    """ Technically legitimate (but odd-looking) paths """
    self.f.create_group('x/y/z')
    self.f['dset'] = 42
    self.assertIn('/', self.f)
    self.assertIn('//', self.f)
    self.assertIn('///', self.f)
    self.assertIn('.///', self.f)
    self.assertIn('././/', self.f)
    grp = self.f['x']
    self.assertIn('.//x/y/z', self.f)
    self.assertNotIn('.//x/y/z', grp)
    self.assertIn('x///', self.f)
    self.assertIn('./x///', self.f)
    self.assertIn('dset///', self.f)
    self.assertIn('/dset//', self.f)