from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize(('weights', 'expected'), (([0.25, 0.05, 0.15, 0.25, 0.15, 0.1, 0.05], [1.554595, 2.463784, 3.0, 3.518378]), ([0.05, 0.05, 0.1, 0.15, 0.15, 0.25, 0.25], [2.84, 3.632973, 4.076216, 4.523243])))
def test_weighted_quantile_no_nan(weights, expected):
    da = DataArray([1, 1.9, 2.2, 3, 3.7, 4.1, 5])
    q = [0.2, 0.4, 0.6, 0.8]
    weights = DataArray(weights)
    expected = DataArray(expected, coords={'quantile': q})
    result = da.weighted(weights).quantile(q)
    assert_allclose(expected, result)