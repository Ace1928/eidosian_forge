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
def test_readonly(self):
    """ Core driver can be used to open existing files """
    fname = self.mktemp()
    fid = File(fname, 'w')
    fid.create_group('foo')
    fid.close()
    fid = File(fname, 'r', driver='core')
    self.assertTrue(fid)
    assert 'foo' in fid
    with self.assertRaises(ValueError):
        fid.create_group('bar')
    fid.close()