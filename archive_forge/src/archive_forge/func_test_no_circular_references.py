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
def test_no_circular_references(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions['x'] = 2
        ds.dimensions['y'] = 2
    gc.collect()
    with h5netcdf.File(tmp_local_netcdf, 'r') as ds:
        refs = gc.get_referrers(ds)
        for ref in refs:
            print(ref)
        assert len(refs) == 1