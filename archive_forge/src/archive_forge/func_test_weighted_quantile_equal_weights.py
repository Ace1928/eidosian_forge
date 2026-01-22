from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('da', (pytest.param([1, 1.9, 2.2, 3, 3.7, 4.1, 5], id='nonan'), pytest.param([1, 1.9, 2.2, 3, 3.7, 4.1, np.nan], id='singlenan'), pytest.param([np.nan, np.nan, np.nan], id='allnan', marks=pytest.mark.filterwarnings('ignore:All-NaN slice encountered:RuntimeWarning'))))
@pytest.mark.parametrize('q', (0.5, (0.2, 0.8)))
@pytest.mark.parametrize('skipna', (True, False))
@pytest.mark.parametrize('factor', [1, 3.14])
def test_weighted_quantile_equal_weights(da: list[float], q: float | tuple[float, ...], skipna: bool, factor: float) -> None:
    data = DataArray(da)
    weights = xr.full_like(data, factor)
    expected = data.quantile(q, skipna=skipna)
    result = data.weighted(weights).quantile(q, skipna=skipna)
    assert_allclose(expected, result)