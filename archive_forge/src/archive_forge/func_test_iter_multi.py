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
def test_iter_multi(self):
    """ External argument may be an iterable of multiple tuples """
    ext_file = self.mktemp()
    N = 100
    external = iter(((ext_file, x * 1000, 1000) for x in range(N)))
    dset = self.f.create_dataset('poo', (6, 100), external=external)
    assert len(dset.external) == N