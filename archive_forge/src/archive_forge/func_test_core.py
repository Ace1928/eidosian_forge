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
def test_core(self):
    """ Core driver is supported (no backing store) """
    fname = self.mktemp()
    fid = File(fname, 'w', driver='core', backing_store=False)
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'core')
    fid.close()
    self.assertFalse(os.path.exists(fname))
    fid = File(self.mktemp(), 'a', driver='core')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'core')
    fid.close()