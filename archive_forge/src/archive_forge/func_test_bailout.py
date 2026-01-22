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
def test_bailout(self):
    """ Returning a non-None value immediately aborts iteration """
    x = self.f.visit(lambda x: x)
    self.assertEqual(x, self.groups[0])
    x = self.f.visititems(lambda x, y: (x, y))
    self.assertEqual(x, (self.groups[0], self.f[self.groups[0]]))