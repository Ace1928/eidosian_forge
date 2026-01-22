from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_cftime
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_cftime_strftime_access(data) -> None:
    """compare cftime formatting against datetime formatting"""
    date_format = '%Y%m%d%H'
    result = data.time.dt.strftime(date_format)
    datetime_array = xr.DataArray(xr.coding.cftimeindex.CFTimeIndex(data.time.values).to_datetimeindex(), name='stftime', coords=data.time.coords, dims=data.time.dims)
    expected = datetime_array.dt.strftime(date_format)
    assert_equal(result, expected)