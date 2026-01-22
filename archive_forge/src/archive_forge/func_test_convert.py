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
def test_convert(self):
    dt = h5py.vlen_dtype(int)
    ds = self.f.create_dataset('vlen', (3,), dtype=dt)
    ds[0] = np.array([1.4, 1.2])
    ds[1] = np.array([1.2])
    ds[2] = [1.2, 2, 3]
    self.assertArrayEqual(ds[0], np.array([1, 1]))
    self.assertArrayEqual(ds[1], np.array([1]))
    self.assertArrayEqual(ds[2], np.array([1, 2, 3]))
    ds[0:2] = np.array([[0.1, 1.1, 2.1, 3.1, 4], np.arange(4)], dtype=object)
    self.assertArrayEqual(ds[0], np.arange(5))
    self.assertArrayEqual(ds[1], np.arange(4))
    ds[0:2] = np.array([np.array([0.1, 1.2, 2.2]), np.array([0.2, 1.2, 2.2])])
    self.assertArrayEqual(ds[0], np.arange(3))
    self.assertArrayEqual(ds[1], np.arange(3))