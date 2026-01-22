from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.core import duck_array_ops
from xarray.tests import (
@requires_cftime
def test_coarsen_coords_cftime():
    times = xr.cftime_range('2000', periods=6)
    da = xr.DataArray(range(6), [('time', times)])
    actual = da.coarsen(time=3).mean()
    expected_times = xr.cftime_range('2000-01-02', freq='3D', periods=2)
    np.testing.assert_array_equal(actual.time, expected_times)