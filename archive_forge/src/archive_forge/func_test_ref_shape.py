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
def test_ref_shape(self):
    """ Region reference shape and selection shape """
    slic = np.s_[25:35, 10:100:5]
    ref = self.dset.regionref[slic]
    self.assertEqual(self.dset.regionref.shape(ref), self.dset.shape)
    self.assertEqual(self.dset.regionref.selection(ref), (10, 18))