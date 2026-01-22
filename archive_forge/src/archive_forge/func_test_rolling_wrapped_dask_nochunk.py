from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('center', (True, None))
def test_rolling_wrapped_dask_nochunk(self, center) -> None:
    pytest.importorskip('dask.array')
    da_day_clim = xr.DataArray(np.arange(1, 367), coords=[np.arange(1, 367)], dims='dayofyear')
    expected = da_day_clim.rolling(dayofyear=31, center=center).mean()
    actual = da_day_clim.chunk().rolling(dayofyear=31, center=center).mean()
    assert_allclose(actual, expected)