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
def test_reopen_file_different_dimension_sizes(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as f:
        f.create_variable('/one/foo', data=[1], dimensions=('x',))
    with h5netcdf.File(tmp_local_netcdf, 'a') as f:
        f.create_variable('/two/foo', data=[1, 2], dimensions=('x',))
    with netCDF4.Dataset(tmp_local_netcdf, 'r') as f:
        assert f.groups['one'].variables['foo'][...].shape == (1,)