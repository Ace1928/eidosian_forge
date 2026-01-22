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
def test_reading_str_array_from_netCDF4(tmp_local_netcdf, decode_vlen_strings):
    with netCDF4.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('foo1', _string_array.shape[0])
        ds.createDimension('foo2', _string_array.shape[1])
        ds.createVariable('bar', str, ('foo1', 'foo2'))
        ds.variables['bar'][:] = _string_array
    ds = h5netcdf.File(tmp_local_netcdf, 'r', **decode_vlen_strings)
    v = ds.variables['bar']
    if getattr(ds, 'decode_vlen_strings', True):
        assert array_equal(v, _string_array)
    else:
        assert array_equal(v, np.char.encode(_string_array))
    ds.close()