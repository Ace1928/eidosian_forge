from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def test_weighted_sum_of_weights_bool():
    da = DataArray([1, 2])
    weights = DataArray([True, True])
    result = da.weighted(weights).sum_of_weights()
    expected = DataArray(2)
    assert_equal(expected, result)