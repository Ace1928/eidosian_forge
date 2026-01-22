import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_method_vanish(self):
    fileobj = io.BytesIO()
    f = h5py.File(fileobj, 'w')
    f.create_dataset('test', data=list(range(12)))
    self.assertEqual(list(f['test'][:]), list(range(12)))
    fileobj.readinto = None
    self.assertRaises(Exception, list, f['test'])