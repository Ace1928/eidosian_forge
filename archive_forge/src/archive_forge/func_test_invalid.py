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
def test_invalid(self):
    """ Test with invalid external lists """
    shape = (6, 100)
    ext_file = self.mktemp()
    for exc_type, external in [(TypeError, [ext_file]), (TypeError, [ext_file, 0]), (TypeError, [ext_file, 0, h5f.UNLIMITED]), (ValueError, [(ext_file,)]), (ValueError, [(ext_file, 0)]), (ValueError, [(ext_file, 0, h5f.UNLIMITED, 0)]), (TypeError, [(ext_file, 0, 'h5f.UNLIMITED')])]:
        with self.assertRaises(exc_type):
            self.f.create_dataset('foo', shape, external=external)