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
def test_creating_variables_with_unlimited_dimensions(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'w') as f:
        f.dimensions['x'] = None
        f.dimensions['y'] = 2
        f.create_variable('dummy', dimensions=('x', 'y'), dtype=np.int64)
        assert f.variables['dummy'].shape == (0, 2)
        assert f.variables['dummy']._h5ds.maxshape == (None, 2)
        with pytest.raises(ValueError) as e:
            f.create_variable('dummy2', data=np.array([[1, 2], [3, 4]]), dimensions=('x', 'y'))
        assert e.value.args[0] == 'Shape tuple is incompatible with data'
        f.create_variable('x', dimensions=('x',), dtype=np.int64)
        assert f.variables['dummy'].shape == (0, 2)
        f.resize_dimension('x', 3)
        assert f.dimensions['x'].size == 3
        np.testing.assert_allclose(f.variables['dummy'], np.zeros((3, 2)))
        f.create_variable('dummy3', dimensions=('x', 'y'), dtype=np.int64)
        assert f.variables['dummy3'].shape == (3, 2)
        assert f.variables['dummy3']._h5ds.maxshape == (None, 2)
        np.testing.assert_allclose(f.variables['dummy3'], np.zeros((3, 2)))
        if tmp_local_or_remote_netcdf.startswith(remote_h5):
            expected_errors = memoryview(b'')
        else:
            expected_errors = pytest.raises(TypeError)
        with expected_errors as e:
            f.variables['dummy3'][:] = np.ones((5, 2))
        if not tmp_local_or_remote_netcdf.startswith(remote_h5):
            assert e.value.args[0] == "Can't broadcast (5, 2) -> (3, 2)"
        assert f.variables['dummy3'].shape == (3, 2)
        assert f.variables['dummy3']._h5ds.maxshape == (None, 2)
        assert f['x'].shape == (3,)
        assert f.dimensions['x'].size == 3
        if tmp_local_or_remote_netcdf.startswith(remote_h5):
            np.testing.assert_allclose(f.variables['dummy3'], np.ones((3, 2)))
        else:
            np.testing.assert_allclose(f.variables['dummy3'], np.zeros((3, 2)))
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r') as f:
        assert f.dimensions['x'].isunlimited()
        assert f.dimensions['x'].size == 3
        assert f._h5file['x'].maxshape == (None,)
        assert f._h5file['x'].shape == (3,)
        assert f.dimensions['y'].size == 2
        assert f._h5file['y'].maxshape == (2,)
        assert f._h5file['y'].shape == (2,)