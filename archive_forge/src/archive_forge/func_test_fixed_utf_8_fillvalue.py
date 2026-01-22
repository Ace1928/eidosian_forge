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
def test_fixed_utf_8_fillvalue(self):
    """ Vlen unicode dataset handles fillvalue """
    dt = h5py.string_dtype(encoding='utf-8', length=10)
    fill_value = 'bár'.encode('utf-8')
    ds = self.f.create_dataset('x', (100,), dtype=dt, fillvalue=fill_value)
    self.assertEqual(self.f['x'][0], fill_value)
    self.assertEqual(self.f['x'].asstr()[0], fill_value.decode('utf-8'))
    self.assertEqual(self.f['x'].fillvalue, fill_value)