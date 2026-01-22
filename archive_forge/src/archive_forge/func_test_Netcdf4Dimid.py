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
def test_Netcdf4Dimid(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as f:
        f.dimensions['x'] = 1
        g = f.create_group('foo')
        g.dimensions['x'] = 2
        g.dimensions['y'] = 3
    with h5py.File(tmp_local_netcdf, 'r') as f:
        dim_ids = {f[name].attrs['_Netcdf4Dimid'] for name in ['x', 'foo/x', 'foo/y']}
        assert dim_ids == {0, 1, 2}