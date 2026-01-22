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
def test_multiple(self):
    """ Opening with two libver args """
    f = File(self.mktemp(), 'w', libver=('earliest', 'v108'))
    self.assertEqual(f.libver, ('earliest', 'v108'))
    f.close()