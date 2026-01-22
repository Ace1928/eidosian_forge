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
def create_netcdf_dimensions(ds, idx):
    g = ds.createGroup('dimtest' + str(idx))
    g.createDimension('time', 0)
    g.createDimension('nvec', 5 + idx)
    g.createDimension('sample', 2 + idx)
    g.createDimension('ship', 3 + idx)
    g.createDimension('ship_strlen', 10)
    g.createDimension('collide', 7 + idx)
    time = g.createVariable('time', 'f8', ('time',))
    data = g.createVariable('data', 'i8', ('ship', 'sample', 'time', 'nvec'))
    collide = g.createVariable('collide', 'i8', ('nvec',))
    non_collide = g.createVariable('non_collide', 'i8', ('nvec',))
    ship = g.createVariable('ship', 'S1', ('ship', 'ship_strlen'))
    sample = g.createVariable('sample', 'i8', ('time', 'sample'))
    time[:] = np.arange(10 + idx)
    data[:] = np.ones((3 + idx, 2 + idx, 10 + idx, 5 + idx)) * 12.0
    collide[...] = np.arange(5 + idx)
    non_collide[...] = np.arange(5 + idx) + 10
    sample[0:2 + idx, :2 + idx] = np.ones((2 + idx, 2 + idx))
    ship[0] = list('Skiff     ')