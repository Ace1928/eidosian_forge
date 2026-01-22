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
def test_mode(self):
    """ Retrieved File objects have a meaningful mode attribute """
    hfile = File(self.mktemp(), 'w')
    try:
        grp = hfile.create_group('foo')
        self.assertEqual(grp.file.mode, hfile.mode)
    finally:
        hfile.close()