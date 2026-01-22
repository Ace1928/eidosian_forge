from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def test_weighted_mean_bool():
    da = DataArray([1, 1])
    weights = DataArray([True, True])
    expected = DataArray(1)
    result = da.weighted(weights).mean()
    assert_equal(expected, result)