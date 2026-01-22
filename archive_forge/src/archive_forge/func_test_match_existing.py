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
def test_match_existing(self):
    """ User block size must match that of file when opening for append """
    name = self.mktemp()
    f = File(name, 'w', userblock_size=512)
    f.close()
    with self.assertRaises(ValueError):
        f = File(name, 'a', userblock_size=1024)
    f = File(name, 'a', userblock_size=512)
    try:
        self.assertEqual(f.userblock_size, 512)
    finally:
        f.close()