import gc
import io
import random
import re
import string
import tempfile
from os import environ as env
import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises
import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError
def test_invalid_netcdf_overwrite_valid(tmp_local_netcdf):
    with netCDF4.Dataset(tmp_local_netcdf, mode='w'):
        pass
    with pytest.warns(UserWarning):
        with h5netcdf.File(tmp_local_netcdf, 'a', invalid_netcdf=True) as f:
            f.create_variable('lzf_compressed', data=[1], dimensions='x', compression='lzf')
            f.create_variable('complex', data=1j)
            f.attrs['complex_attr'] = 1j
            f.create_variable('scaleoffset', data=[1], dimensions=('x',), scaleoffset=0)
    with h5netcdf.File(tmp_local_netcdf, 'r') as f:
        np.testing.assert_equal(f['lzf_compressed'][:], [1])
        assert f['complex'][...] == 1j
        assert f.attrs['complex_attr'] == 1j
        np.testing.assert_equal(f['scaleoffset'][:], [1])
    h5 = get_hdf5_module(tmp_local_netcdf)
    with h5.File(tmp_local_netcdf, 'r') as f:
        assert '_NCProperties' not in f.attrs