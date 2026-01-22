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
def test_numpy_commutative(self):
    """
        Create a h5py dataset, extract one element convert to numpy
        Check that it returns symmetric response to == and !=
        """
    shape = (100, 1)
    dset = self.f.create_dataset('test', shape, dtype=float, data=np.random.rand(*shape))
    val = np.float64(dset[0])
    assert np.all((val == dset) == (dset == val))
    assert np.all((val != dset) == (dset != val))
    delta = 0.001
    nval = np.nanmax(dset) + delta
    assert np.all((nval == dset) == (dset == nval))
    assert np.all((nval != dset) == (dset != nval))