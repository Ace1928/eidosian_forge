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
def test_backing(self):
    """ Core driver saves to file when backing store used """
    fname = self.mktemp()
    fid = File(fname, 'w', driver='core', backing_store=True)
    fid.create_group('foo')
    fid.close()
    fid = File(fname, 'r')
    assert 'foo' in fid
    fid.close()
    with self.assertRaises(TypeError):
        File(fname, 'w', backing_store=True)