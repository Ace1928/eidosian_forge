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
def write_dimensions(tmp_netcdf, write_module):
    if write_module in [legacyapi, netCDF4]:
        with write_module.Dataset(tmp_netcdf, 'w') as ds:
            create_netcdf_dimensions(ds, 0)
            create_netcdf_dimensions(ds, 1)
    else:
        with write_module.File(tmp_netcdf, 'w') as ds:
            create_h5netcdf_dimensions(ds, 0)
            create_h5netcdf_dimensions(ds, 1)