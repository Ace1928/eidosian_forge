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
def test_unicode_hdf5_path(self):
    """
        Check that external links handle unicode hdf5 paths properly
        Testing issue #333
        """
    ext_filename = os.path.join(mkdtemp(), 'external.hdf5')
    with File(ext_filename, 'w') as ext_file:
        ext_file.create_group('α')
        ext_file['α'].attrs['ext_attr'] = 'test'
    self.f['ext'] = ExternalLink(ext_filename, '/α')
    self.assertEqual(self.f['ext'].attrs['ext_attr'], 'test')