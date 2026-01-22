from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('center', (True, False))
@pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
@pytest.mark.parametrize('window', (1, 2, 3, 4))
@pytest.mark.parametrize('name', ('sum', 'max'))
def test_rolling_reduce_nonnumeric(self, center, min_periods, window, name, compute_backend) -> None:
    da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims='time').isnull()
    if min_periods is not None and window < min_periods:
        min_periods = window
    rolling_obj = da.rolling(time=window, center=center, min_periods=min_periods)
    actual = rolling_obj.reduce(getattr(np, 'nan%s' % name))
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    assert actual.sizes == expected.sizes