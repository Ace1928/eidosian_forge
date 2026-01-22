from h5py import File
from h5py._hl.base import is_hdf5, Empty
from .common import ut, TestCase, UNICODE_FILENAMES
import numpy as np
import os
import tempfile
def test_is_hdf5():
    filename = File(tempfile.mktemp(), 'w').filename
    assert is_hdf5(filename)
    filename = tempfile.mktemp()
    assert not is_hdf5(filename)