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
def test_reading_unused_unlimited_dimension(tmp_local_or_remote_netcdf):
    """Test reading a file with unused dimension of unlimited size"""
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'w') as f:
        f.dimensions = {'x': None}
        f.resize_dimension('x', 5)
        assert f.dimensions['x'].isunlimited()
        assert f.dimensions['x'].size == 5