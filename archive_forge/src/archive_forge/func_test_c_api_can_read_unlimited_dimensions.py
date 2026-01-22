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
def test_c_api_can_read_unlimited_dimensions(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as f:
        f.dimensions['x'] = None
        f.dimensions['y'] = 3
        f.dimensions['z'] = None
        f.create_variable('dummy1', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummy2', dimensions=('y', 'x', 'x'), dtype=np.int64)
        g = f.create_group('test')
        g.create_variable('dummy3', dimensions=('y', 'y'), dtype=np.int64)
        g.create_variable('dummy4', dimensions=('z', 'z'), dtype=np.int64)
        f.resize_dimension('x', 2)
    with netCDF4.Dataset(tmp_local_netcdf, 'r') as f:
        assert f.dimensions['x'].size == 2
        assert f.dimensions['x'].isunlimited() is True
        assert f.dimensions['y'].size == 3
        assert f.dimensions['y'].isunlimited() is False
        assert f.dimensions['z'].size == 0
        assert f.dimensions['z'].isunlimited() is True
        assert f.variables['dummy1'].shape == (2, 3)
        assert f.variables['dummy2'].shape == (3, 2, 2)
        g = f.groups['test']
        assert g.variables['dummy3'].shape == (3, 3)
        assert g.variables['dummy4'].shape == (0, 0)