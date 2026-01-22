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
def test_invalid_netcdf4(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip('netCDF4 package does not work with remote HDF5 files')
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, 'w') as f:
        var, var2 = create_invalid_netcdf_data()
        grps = ['bar', 'baz']
        for grp in grps:
            fx = f.create_group(grp)
            for k, v in var.items():
                fx.create_dataset(k, data=v)
            for k, v in var2.items():
                fx.create_dataset(k, data=np.arange(v))
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r', phony_dims='sort') as dsr:
        for i, grp in enumerate(grps):
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r', phony_dims='access') as dsr:
        for i, grp in enumerate(grps):
            var = dsr[grp].variables
            check_invalid_netcdf4(var, i)
    if not tmp_local_or_remote_netcdf.startswith(remote_h5):
        with netCDF4.Dataset(tmp_local_or_remote_netcdf, 'r') as dsr:
            for i, grp in enumerate(grps):
                var = dsr[grp].variables
                check_invalid_netcdf4(var, i)
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r') as ds:
        with raises(ValueError):
            ds['bar'].variables['foo1'].dimensions
    with raises(ValueError):
        with h5netcdf.File(tmp_local_or_remote_netcdf, 'r', phony_dims='srt') as ds:
            pass