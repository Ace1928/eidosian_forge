from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def test_weighted_quantile_bool():
    da = DataArray([1, 1])
    weights = DataArray([True, True])
    q = 0.5
    expected = DataArray([1], coords={'quantile': [q]}).squeeze()
    result = da.weighted(weights).quantile(q)
    assert_equal(expected, result)