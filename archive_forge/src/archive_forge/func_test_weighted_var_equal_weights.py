from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('da', ([1.0, 2], [1, np.nan]))
@pytest.mark.parametrize('skipna', (True, False))
@pytest.mark.parametrize('factor', [1, 2, 3.14])
def test_weighted_var_equal_weights(da, skipna, factor):
    da = DataArray(da)
    weights = xr.full_like(da, factor)
    expected = da.var(skipna=skipna)
    result = da.weighted(weights).var(skipna=skipna)
    assert_equal(expected, result)