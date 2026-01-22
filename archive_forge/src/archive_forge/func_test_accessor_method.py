from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@pytest.mark.parametrize('method, parameters', [('floor', 'D'), ('ceil', 'D'), ('round', 'D')])
def test_accessor_method(self, method, parameters) -> None:
    dates = pd.date_range('2014-01-01', '2014-05-01', freq='h')
    xdates = xr.DataArray(dates, dims=['time'])
    expected = getattr(dates, method)(parameters)
    actual = getattr(xdates.dt, method)(parameters)
    assert_array_equal(expected, actual)