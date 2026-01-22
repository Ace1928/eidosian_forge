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
@pytest.mark.skipif(version.parse(h5py.__version__) < version.parse('3.7.0'), reason='h5py<3.7.0 bug with track_order prevents editing with netCDF4')
def test_creation_with_h5netcdf_edit_with_netcdf4(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as the_file:
        the_file.dimensions = {'x': 5}
        variable = the_file.create_variable('hello', ('x',), float)
        variable[...] = 5
    with netCDF4.Dataset(tmp_local_netcdf, mode='a') as the_file:
        variable = the_file['hello']
        np.testing.assert_array_equal(variable[...].data, 5)
        variable[:3] = 2
        variable = the_file.createVariable('goodbye', float, ('x',))
        variable[...] = 10
    with h5netcdf.File(tmp_local_netcdf, 'a') as the_file:
        variable = the_file['hello']
        np.testing.assert_array_equal(variable[...].data, [2, 2, 2, 5, 5])
        variable = the_file['goodbye']
        np.testing.assert_array_equal(variable[...].data, 10)