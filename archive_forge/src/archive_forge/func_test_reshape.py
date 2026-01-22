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
def test_reshape(self):
    """ Create from existing data, and make it fit a new shape """
    data = np.arange(30, dtype='f')
    dset = self.f.create_dataset('foo', shape=(10, 3), data=data)
    self.assertEqual(dset.shape, (10, 3))
    self.assertArrayEqual(dset[...], data.reshape((10, 3)))