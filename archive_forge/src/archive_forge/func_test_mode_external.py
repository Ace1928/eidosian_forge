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
def test_mode_external(self):
    """ Mode property works for files opened via external links

        Issue 190.
        """
    fname1 = self.mktemp()
    fname2 = self.mktemp()
    f1 = File(fname1, 'w')
    f1.close()
    f2 = File(fname2, 'w')
    try:
        f2['External'] = h5py.ExternalLink(fname1, '/')
        f3 = f2['External'].file
        self.assertEqual(f3.mode, 'r+')
    finally:
        f2.close()
        f3.close()
    f2 = File(fname2, 'r')
    try:
        f3 = f2['External'].file
        self.assertEqual(f3.mode, 'r')
    finally:
        f2.close()
        f3.close()