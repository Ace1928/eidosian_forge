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
def test_invalid_netcdf_error(tmp_local_or_remote_netcdf):
    if tmp_local_or_remote_netcdf.startswith(remote_h5):
        pytest.skip('Remote HDF5 does not yet support LZF compression')
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'w', invalid_netcdf=False) as f:
        f.create_variable('lzf_compressed', data=[1], dimensions='x', compression='lzf')
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable('complex', data=1j)
        with pytest.raises(h5netcdf.CompatibilityError):
            f.attrs['complex_attr'] = 1j
        with pytest.raises(h5netcdf.CompatibilityError):
            f.create_variable('scaleoffset', data=[1], dimensions=('x',), scaleoffset=0)