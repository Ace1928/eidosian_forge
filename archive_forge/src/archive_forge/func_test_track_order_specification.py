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
def test_track_order_specification(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w', track_order=False):
        pass
    with h5netcdf.File(tmp_local_netcdf, 'w', track_order=True):
        pass