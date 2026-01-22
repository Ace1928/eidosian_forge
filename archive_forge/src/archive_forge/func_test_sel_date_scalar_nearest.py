from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
@pytest.mark.parametrize('sel_kwargs', [{'method': 'nearest'}, {'method': 'nearest', 'tolerance': timedelta(days=70)}, {'method': 'nearest', 'tolerance': timedelta(days=1800000)}])
def test_sel_date_scalar_nearest(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(2).assign_coords(time=index[1])
    result = da.sel(time=date_type(1, 4, 1), **sel_kwargs)
    assert_identical(result, expected)
    expected = xr.DataArray(3).assign_coords(time=index[2])
    result = da.sel(time=date_type(1, 11, 1), **sel_kwargs)
    assert_identical(result, expected)