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
def write_legacy_netcdf(tmp_netcdf, write_module):
    ds = write_module.Dataset(tmp_netcdf, 'w')
    ds.setncattr('global', 42)
    ds.other_attr = 'yes'
    ds.createDimension('x', 4)
    ds.createDimension('y', 5)
    ds.createDimension('z', 6)
    ds.createDimension('empty', 0)
    ds.createDimension('string3', 3)
    ds.createDimension('unlimited', None)
    v = ds.createVariable('foo', float, ('x', 'y'), chunksizes=(4, 5), zlib=True)
    v[...] = 1
    v.setncattr('units', 'meters')
    v = ds.createVariable('y', int, ('y',), fill_value=-1)
    v[:4] = np.arange(4)
    v = ds.createVariable('z', 'S1', ('z', 'string3'), fill_value=b'X')
    v[...] = _char_array
    v = ds.createVariable('scalar', np.float32, ())
    v[...] = 2.0
    v = ds.createVariable('intscalar', np.int64, (), zlib=6, fill_value=None)
    v[...] = 2
    v = ds.createVariable('foo_unlimited', float, ('x', 'unlimited'))
    v[...] = 1
    with raises((h5netcdf.CompatibilityError, TypeError)):
        ds.createVariable('boolean', np.bool_, 'x')
    g = ds.createGroup('subgroup')
    v = g.createVariable('subvar', np.int32, ('x',))
    v[...] = np.arange(4.0)
    g.createDimension('y', 10)
    g.createVariable('y_var', float, ('y',))
    ds.createDimension('mismatched_dim', 1)
    ds.createVariable('mismatched_dim', int, ())
    v = ds.createVariable('var_len_str', str, 'x')
    v[0] = 'foo'
    ds.close()