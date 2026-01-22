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
def test_reuse_struct_from_other(self):
    dt = [('a', int), ('b', h5py.vlen_dtype(int))]
    ds = self.f.create_dataset('vlen', (1,), dtype=dt)
    fname = self.f.filename
    self.f.close()
    self.f = h5py.File(fname, 'a')
    self.f.create_dataset('vlen2', (1,), self.f['vlen']['b'][()].dtype)