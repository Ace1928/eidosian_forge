from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def test_identical_coords_no_computes():
    lons2 = xr.DataArray(da.zeros((10, 10), chunks=2), dims=('y', 'x'))
    a = xr.DataArray(da.zeros((10, 10), chunks=2), dims=('y', 'x'), coords={'lons': lons2})
    b = xr.DataArray(da.zeros((10, 10), chunks=2), dims=('y', 'x'), coords={'lons': lons2})
    with raise_if_dask_computes():
        c = a + b
    assert_identical(c, a)