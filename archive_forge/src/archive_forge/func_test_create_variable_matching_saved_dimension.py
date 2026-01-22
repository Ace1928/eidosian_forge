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
def test_create_variable_matching_saved_dimension(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'w') as f:
        f.dimensions['x'] = 2
        f.create_variable('y', data=[1, 2], dimensions=('x',))
    with h5.File(tmp_local_or_remote_netcdf, 'r') as f:
        dimlen = f'{f['y'].dims[0].values()[0].size:10}'
        assert f['y'].dims[0].keys() == [NOT_A_VARIABLE.decode('ascii') + dimlen]
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'a') as f:
        f.create_variable('x', data=[0, 1], dimensions=('x',))
    with h5.File(tmp_local_or_remote_netcdf, 'r') as f:
        assert f['y'].dims[0].keys() == ['x']