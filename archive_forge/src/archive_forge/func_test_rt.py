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
def test_rt(self):
    """ Compound types are read back in correct order (issue 236)"""
    dt = np.dtype([('weight', np.float64), ('cputime', np.float64), ('walltime', np.float64), ('parents_offset', np.uint32), ('n_parents', np.uint32), ('status', np.uint8), ('endpoint_type', np.uint8)])
    testdata = np.ndarray((16,), dtype=dt)
    for key in dt.fields:
        testdata[key] = np.random.random((16,)) * 100
    self.f['test'] = testdata
    outdata = self.f['test'][...]
    self.assertTrue(np.all(outdata == testdata))
    self.assertEqual(outdata.dtype, testdata.dtype)