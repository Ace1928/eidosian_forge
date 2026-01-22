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
@ut.skipUnless(direct_vfd, 'DIRECT driver is supported on Linux if hdf5 is built with the appriorate flags.')
def test_direct(self):
    """ DIRECT driver is supported on Linux"""
    fid = File(self.mktemp(), 'w', driver='direct')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'direct')
    default_fapl = fid.id.get_access_plist().get_fapl_direct()
    fid.close()
    fid = File(self.mktemp(), 'a', driver='direct')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'direct')
    fid.close()
    for alignment, block_size, cbuf_size in [default_fapl, (default_fapl[0], default_fapl[1], 3 * default_fapl[1]), (default_fapl[0] * 2, default_fapl[1], 3 * default_fapl[1]), (default_fapl[0], 2 * default_fapl[1], 6 * default_fapl[1])]:
        with File(self.mktemp(), 'w', driver='direct', alignment=alignment, block_size=block_size, cbuf_size=cbuf_size) as fid:
            actual_fapl = fid.id.get_access_plist().get_fapl_direct()
            actual_alignment = actual_fapl[0]
            actual_block_size = actual_fapl[1]
            actual_cbuf_size = actual_fapl[2]
            assert actual_alignment == alignment
            assert actual_block_size == block_size
            assert actual_cbuf_size == actual_cbuf_size