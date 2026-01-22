import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
@pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 14, 4), reason='Requires HDF5 >= 1.14.4')
def test_open_nonpage_pbs(self):
    """Open non-PAGE file with page buffer set."""
    fname = self.mktemp()
    fsp = 16 * 1024
    with File(fname, mode='w'):
        pass
    with File(fname, mode='r', page_buf_size=fsp) as f:
        fapl = f.id.get_access_plist()
        assert fapl.get_page_buffer_size()[0] == 0