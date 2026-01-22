from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def test_weighted_quantile_zero_weights():
    da = DataArray([0, 1, 2, 3])
    weights = DataArray([1, 0, 1, 0])
    q = 0.75
    result = da.weighted(weights).quantile(q)
    expected = DataArray([0, 2]).quantile(0.75)
    assert_allclose(expected, result)