from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('center', (True, False))
@pytest.mark.parametrize('window', (1, 2, 3, 4))
def test_rolling_construct_stride(self, center: bool, window: int) -> None:
    df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 'time': np.linspace(0, 1, 20)})
    ds = Dataset.from_dataframe(df)
    df_rolling_mean = df.rolling(window, center=center, min_periods=1).mean()
    ds_rolling = ds.rolling(index=window, center=center)
    ds_rolling_mean = ds_rolling.construct('w', stride=2).mean('w')
    np.testing.assert_allclose(df_rolling_mean['x'][::2].values, ds_rolling_mean['x'].values)
    np.testing.assert_allclose(df_rolling_mean.index[::2], ds_rolling_mean['index'])
    ds2 = ds.drop_vars('index')
    ds2_rolling = ds2.rolling(index=window, center=center)
    ds2_rolling_mean = ds2_rolling.construct('w', stride=2).mean('w')
    np.testing.assert_allclose(df_rolling_mean['x'][::2].values, ds2_rolling_mean['x'].values)
    ds3 = xr.Dataset({'x': ('t', range(20)), 'x2': ('y', range(5))}, {'t': range(20), 'y': ('y', range(5)), 't2': ('t', range(20)), 'y2': ('y', range(5)), 'yt': (['t', 'y'], np.ones((20, 5)))})
    ds3_rolling = ds3.rolling(t=window, center=center)
    ds3_rolling_mean = ds3_rolling.construct('w', stride=2).mean('w')
    for coord in ds3.coords:
        assert coord in ds3_rolling_mean.coords