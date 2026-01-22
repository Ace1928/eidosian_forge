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
def test_too_small_pbs(self):
    """Page buffer size must be greater than file space page size."""
    fname = self.mktemp()
    fsp = 16 * 1024
    with File(fname, mode='w', fs_strategy='page', fs_page_size=fsp):
        pass
    with self.assertRaises(OSError):
        File(fname, mode='r', page_buf_size=fsp - 1)