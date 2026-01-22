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
def test_scales_on_append(tmp_local_netcdf):
    with netCDF4.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('x', 10)
    with netCDF4.Dataset(tmp_local_netcdf, 'r+') as ds:
        ds.createVariable('test', 'i4', ('x',))
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        assert ds.variables['test'].attrs._h5attrs.get('DIMENSION_LIST', False)
    with legacyapi.Dataset(tmp_local_netcdf, 'r+') as ds:
        ds.createVariable('test1', 'i4', ('x',))
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        assert ds.variables['test1'].attrs._h5attrs.get('DIMENSION_LIST', False)