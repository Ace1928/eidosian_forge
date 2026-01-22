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
@ut.skipUnless(os.name == 'posix', 'Stdio driver is supported on posix')
def test_stdio(self):
    """ Stdio driver is supported on posix """
    fid = File(self.mktemp(), 'w', driver='stdio')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'stdio')
    fid.close()
    fid = File(self.mktemp(), 'a', driver='stdio')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'stdio')
    fid.close()