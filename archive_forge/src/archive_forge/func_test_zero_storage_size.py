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
def test_zero_storage_size():
    from io import BytesIO
    buf = BytesIO()
    with h5py.File(buf, 'w') as fout:
        fout.create_dataset('empty', dtype='uint8')
    buf.seek(0)
    with h5py.File(buf, 'r') as fin:
        assert fin['empty'].chunks is None
        assert fin['empty'].id.get_offset() is None
        assert fin['empty'].id.get_storage_size() == 0