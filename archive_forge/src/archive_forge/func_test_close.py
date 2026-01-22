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
def test_close(self):
    """ Closing a file invalidates any of the file's open objects """
    with File(self.mktemp(), 'w') as f1:
        g1 = f1.create_group('foo')
        self.assertTrue(bool(f1.id))
        self.assertTrue(bool(g1.id))
        f1.close()
        self.assertFalse(bool(f1.id))
        self.assertFalse(bool(g1.id))
    with File(self.mktemp(), 'w') as f2:
        g2 = f2.create_group('foo')
        self.assertTrue(bool(f2.id))
        self.assertTrue(bool(g2.id))
        self.assertFalse(bool(f1.id))
        self.assertFalse(bool(g1.id))