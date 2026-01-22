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
def test_create(self):
    """ Creating external links """
    self.f['ext'] = ExternalLink(self.ename, '/external')
    grp = self.f['ext']
    self.ef = grp.file
    self.assertNotEqual(self.ef, self.f)
    self.assertEqual(grp.name, '/external')