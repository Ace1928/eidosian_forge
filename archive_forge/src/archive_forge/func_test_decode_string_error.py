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
def test_decode_string_error(tmp_local_or_remote_netcdf):
    write_h5netcdf(tmp_local_or_remote_netcdf)
    with pytest.raises(TypeError):
        with h5netcdf.legacyapi.Dataset(tmp_local_or_remote_netcdf, 'r', decode_vlen_strings=True) as ds:
            assert ds.name == '/'