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
def test_2d_partial_slice(self):
    dset = self.f.create_dataset('foo', (5, 5), chunks=(2, 2))
    expected = ((slice(3, 4, 1), slice(3, 4, 1)), (slice(3, 4, 1), slice(4, 5, 1)), (slice(4, 5, 1), slice(3, 4, 1)), (slice(4, 5, 1), slice(4, 5, 1)))
    sel = slice(3, 5)
    self.assertEqual(list(dset.iter_chunks((sel, sel))), list(expected))