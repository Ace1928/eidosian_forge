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
def test_h5netcdf_chunking(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 10, 'y': 10, 'z': 10, 't': None}
        v = ds.create_variable('hello', ('x', 'y', 'z', 't'), 'float', chunking_heuristic='h5netcdf')
        chunks_h5netcdf = v.chunks
    assert chunks_h5netcdf == (10, 10, 10, 1)
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 10, 't': None}
        v = ds.create_variable('hello', ('x', 't'), 'float', chunking_heuristic='h5netcdf')
        chunks_h5netcdf = v.chunks
    assert chunks_h5netcdf == (10, 128)
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 10, 'y': 10, 'z': 10, 't': None}
        ds.resize_dimension('t', 10)
        v = ds.create_variable('hello', ('x', 'y', 'z', 't'), 'float', chunking_heuristic='h5netcdf')
        chunks_h5netcdf = v.chunks
    assert chunks_h5netcdf == (5, 5, 5, 10)