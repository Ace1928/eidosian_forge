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
@pytest.fixture(params=[[netCDF4, netCDF4], [legacyapi, legacyapi], [h5netcdf, h5netcdf], [legacyapi, netCDF4], [netCDF4, legacyapi], [h5netcdf, netCDF4], [netCDF4, h5netcdf], [legacyapi, h5netcdf], [h5netcdf, legacyapi]])
def read_write_matrix(request):
    print('write module:', request.param[0].__name__)
    print('read_module:', request.param[1].__name__)
    return request.param