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
def test_shape_is_tied_to_coordinate(tmp_local_or_remote_netcdf):
    with h5netcdf.legacyapi.Dataset(tmp_local_or_remote_netcdf, 'w') as ds:
        ds.createDimension('x', size=None)
        ds.createVariable('xvar', int, ('x',))
        ds['xvar'][:5] = np.arange(5)
        assert ds['xvar'].shape == (5,)
        ds.createVariable('yvar', int, ('x',))
        ds['yvar'][:10] = np.arange(10)
        assert ds['yvar'].shape == (10,)
        assert ds['xvar'].shape == (10,)