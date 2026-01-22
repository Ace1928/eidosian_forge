from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.slow
@pytest.mark.parametrize('ds', (1, 2), indirect=True)
@pytest.mark.parametrize('center', (True, False))
@pytest.mark.parametrize('min_periods', (None, 1, 2, 3))
@pytest.mark.parametrize('window', (1, 2, 3, 4))
@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'var', 'min', 'max', 'median'))
def test_rolling_reduce(self, ds, center, min_periods, window, name) -> None:
    if min_periods is not None and window < min_periods:
        min_periods = window
    if name == 'std' and window == 1:
        pytest.skip('std with window == 1 is unstable in bottleneck')
    rolling_obj = ds.rolling(time=window, center=center, min_periods=min_periods)
    actual = rolling_obj.reduce(getattr(np, 'nan%s' % name))
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    assert ds.sizes == actual.sizes
    assert list(ds.data_vars.keys()) == list(actual.data_vars.keys())
    for key, src_var in ds.data_vars.items():
        assert src_var.dims == actual[key].dims