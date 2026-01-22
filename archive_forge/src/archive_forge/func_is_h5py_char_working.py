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
def is_h5py_char_working(tmp_netcdf, name):
    h5 = get_hdf5_module(tmp_netcdf)
    with h5.File(tmp_netcdf, 'r') as ds:
        v = ds[name]
        try:
            assert array_equal(v, _char_array)
            return True
        except Exception as e:
            if re.match("^Can't read data", e.args[0]):
                return False
            else:
                raise