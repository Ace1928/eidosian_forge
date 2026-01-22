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
def test_rolling_pandas_compat(self, center, window, min_periods) -> None:
    df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 'time': np.linspace(0, 1, 20)})
    ds = Dataset.from_dataframe(df)
    if min_periods is not None and window < min_periods:
        min_periods = window
    df_rolling = df.rolling(window, center=center, min_periods=min_periods).mean()
    ds_rolling = ds.rolling(index=window, center=center, min_periods=min_periods).mean()
    np.testing.assert_allclose(df_rolling['x'].values, ds_rolling['x'].values)
    np.testing.assert_allclose(df_rolling.index, ds_rolling['index'])