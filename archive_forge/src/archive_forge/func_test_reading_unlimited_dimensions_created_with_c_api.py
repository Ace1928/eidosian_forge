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
def test_reading_unlimited_dimensions_created_with_c_api(tmp_local_netcdf):
    with netCDF4.Dataset(tmp_local_netcdf, 'w') as f:
        f.createDimension('x', None)
        f.createDimension('y', 3)
        f.createDimension('z', None)
        dummy1 = f.createVariable('dummy1', float, ('x', 'y'))
        f.createVariable('dummy2', float, ('y', 'x', 'x'))
        g = f.createGroup('test')
        g.createVariable('dummy3', float, ('y', 'y'))
        g.createVariable('dummy4', float, ('z', 'z'))
        dummy1[:] = [[1, 2, 3], [4, 5, 6]]
        f.createVariable('dummy5', float, ('x', 'y'))
    with h5netcdf.File(tmp_local_netcdf, 'r') as f:
        assert f.dimensions['x'].isunlimited()
        assert f.dimensions['y'].size == 3
        assert f.dimensions['z'].isunlimited()
        assert f.dimensions['x'].size == 2
        assert f.dimensions['y'].size == 3
        assert f.dimensions['z'].size == 0
        assert f['dummy2'].shape == (3, 2, 2)
        f.groups['test']['dummy3'].shape == (3, 3)
        f.groups['test']['dummy4'].shape == (0, 0)
        assert f['dummy5'].shape == (2, 3)