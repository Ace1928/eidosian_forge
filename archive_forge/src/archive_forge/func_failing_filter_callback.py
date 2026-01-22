from ctypes import (
import pytest
import h5py
from h5py import h5z
from .common import insubprocess
@H5ZFuncT
def failing_filter_callback(flags, cd_nelemts, cd_values, nbytes, buf_size, buf):
    return 0