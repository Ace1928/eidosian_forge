import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_exception_writeonly(self):
    fileobj = open(os.path.join(self.tempdir, 'a.h5'), 'wb')
    with self.assertRaises(io.UnsupportedOperation):
        f = h5py.File(fileobj, 'w')
        group = f.create_group('group')
        group.create_dataset('data', data='foo', dtype=h5py.string_dtype())