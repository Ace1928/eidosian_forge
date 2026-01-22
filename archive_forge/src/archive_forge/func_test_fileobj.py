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
def test_fileobj(self):
    """ Python file object driver is supported """
    tf = tempfile.TemporaryFile()
    fid = File(tf, 'w', driver='fileobj')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'fileobj')
    fid.close()
    with self.assertRaises(ValueError):
        File(tf, 'w', driver='core')