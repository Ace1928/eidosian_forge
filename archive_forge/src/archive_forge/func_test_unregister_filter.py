from ctypes import (
import pytest
import h5py
from h5py import h5z
from .common import insubprocess
@pytest.mark.mpi_skip
@insubprocess
def test_unregister_filter(request):
    if h5py.h5z.filter_avail(h5py.h5z.FILTER_LZF):
        res = h5py.h5z.unregister_filter(h5py.h5z.FILTER_LZF)
        assert res