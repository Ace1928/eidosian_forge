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
def test_rolling_construct(self, center: bool, window: int) -> None:
    df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 'time': np.linspace(0, 1, 20)})
    ds = Dataset.from_dataframe(df)
    df_rolling = df.rolling(window, center=center, min_periods=1).mean()
    ds_rolling = ds.rolling(index=window, center=center)
    ds_rolling_mean = ds_rolling.construct('window').mean('window')
    np.testing.assert_allclose(df_rolling['x'].values, ds_rolling_mean['x'].values)
    np.testing.assert_allclose(df_rolling.index, ds_rolling_mean['index'])
    ds_rolling_mean = ds_rolling.construct('window', stride=2, fill_value=0.0).mean('window')
    assert (ds_rolling_mean.isnull().sum() == 0).to_dataarray(dim='vars').all()
    assert (ds_rolling_mean['x'] == 0.0).sum() >= 0