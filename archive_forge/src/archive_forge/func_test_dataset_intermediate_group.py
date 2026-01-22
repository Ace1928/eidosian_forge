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
def test_dataset_intermediate_group(self):
    """ Create dataset with missing intermediate groups """
    ds = self.f.create_dataset('/foo/bar/baz', shape=(10, 10), dtype='<i4')
    self.assertIsInstance(ds, h5py.Dataset)
    self.assertTrue('/foo/bar/baz' in self.f)