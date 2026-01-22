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
def test_actual_pbs(self):
    """Verify actual page buffer size."""
    fname = self.mktemp()
    fsp = 16 * 1024
    pbs = 2 * fsp
    with File(fname, mode='w', fs_strategy='page', fs_page_size=fsp):
        pass
    with File(fname, mode='r', page_buf_size=pbs - 1) as f:
        fapl = f.id.get_access_plist()
        self.assertEqual(fapl.get_page_buffer_size()[0], fsp)