import pathlib
import os
import sys
import numpy as np
import platform
import pytest
import warnings
from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace, product
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
from h5py import version
import h5py
import h5py._hl.selections as sel
def test_python_int_uint64(writable_file):
    data = [np.iinfo(np.int64).max, np.iinfo(np.int64).max + 1]
    ds = writable_file.create_dataset('x', data=data, dtype=np.uint64)
    assert ds.dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(ds[:], np.array(data, dtype=np.uint64))
    ds[:] = data
    np.testing.assert_array_equal(ds[:], np.array(data, dtype=np.uint64))