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
def test_unicode_default(self):
    """ Unicode names convertible to ASCII are stored as ASCII (issue 239)
        """
    name = u'/Hello, this is a name'
    group = self.f.create_group(name)
    self.assertEqual(group.name, name)
    self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_ASCII)