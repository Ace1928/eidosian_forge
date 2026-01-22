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
def test_overwrite_existing_file(tmp_local_netcdf):
    with netCDF4.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('x', 10)
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        assert ds.attrs._h5attrs.get('_NCProperties', False)
    with legacyapi.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('x', 10)
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        assert ds.attrs._h5attrs.get('_NCProperties', False)
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions['x'] = 10
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        assert ds.attrs._h5attrs.get('_NCProperties', False)