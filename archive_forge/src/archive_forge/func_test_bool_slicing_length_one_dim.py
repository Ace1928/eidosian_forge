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
def test_bool_slicing_length_one_dim(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 1, 'y': 2}
        v = ds.create_variable('hello', ('x', 'y'), 'float')
        v[:] = np.ones((1, 2))
    bool_slice = np.array([1], dtype=bool)
    with legacyapi.Dataset(tmp_local_netcdf, 'a') as ds:
        data = ds['hello'][bool_slice, :]
        np.testing.assert_equal(data, np.ones((1, 2)))
        ds['hello'][bool_slice, :] = np.zeros((1, 2))
        data = ds['hello'][bool_slice, :]
        np.testing.assert_equal(data, np.zeros((1, 2)))
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        h5py_version = version.parse(h5py.__version__)
        if version.parse('3.0.0') <= h5py_version < version.parse('3.7.0'):
            error = 'Indexing arrays must have integer dtypes'
            with pytest.raises(TypeError) as e:
                ds['hello'][bool_slice, :]
            assert error == str(e.value)
        else:
            ds['hello'][bool_slice, :]