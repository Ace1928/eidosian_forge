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
def create_h5netcdf_dimensions(ds, idx):
    g = ds.create_group('dimtest' + str(idx))
    g.dimensions['time'] = 0
    g.dimensions['nvec'] = 5 + idx
    g.dimensions['sample'] = 2 + idx
    g.dimensions['ship'] = 3 + idx
    g.dimensions['ship_strlen'] = 10
    g.dimensions['collide'] = 7 + idx
    g.create_variable('time', dimensions=('time',), dtype=np.float64)
    g.create_variable('data', dimensions=('ship', 'sample', 'time', 'nvec'), dtype=np.int64)
    g.create_variable('collide', dimensions=('nvec',), dtype=np.int64)
    g.create_variable('non_collide', dimensions=('nvec',), dtype=np.int64)
    g.create_variable('sample', dimensions=('time', 'sample'), dtype=np.int64)
    g.create_variable('ship', dimensions=('ship', 'ship_strlen'), dtype='S1')
    g.resize_dimension('time', 10 + idx)
    g.variables['time'][:] = np.arange(10 + idx)
    g.variables['data'][:] = np.ones((3 + idx, 2 + idx, 10 + idx, 5 + idx)) * 12.0
    g.variables['collide'][...] = np.arange(5 + idx)
    g.variables['non_collide'][...] = np.arange(5 + idx) + 10
    g.variables['sample'][0:2 + idx, :2 + idx] = np.ones((2 + idx, 2 + idx))
    g.variables['ship'][0] = list('Skiff     ')