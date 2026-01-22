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
def create_attach_scales(filename, append_module):
    with netCDF4.Dataset(filename, 'w') as ds:
        ds.createDimension('x', 0)
        ds.createDimension('y', 1)
        ds.createVariable('test', 'i4', ('x',))
        ds.variables['test'] = np.ones((10,))
    with append_module.Dataset(filename, 'a') as ds:
        ds.createVariable('test1', 'i4', ('x',))
        ds.createVariable('y', 'i4', ('x', 'y'))
    with h5netcdf.File(filename, 'r') as ds:
        refs = ds._h5group['x'].attrs.get('REFERENCE_LIST', False)
        assert len(refs) == 3
        for (ref, dim), name in zip(refs, ['/test', '/test1', '/_nc4_non_coord_y']):
            assert dim == 0
            assert ds._root._h5file[ref].name == name