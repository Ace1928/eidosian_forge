from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def check_weighted_operations(data, weights, dim, skipna):
    result = data.weighted(weights).sum_of_weights(dim)
    expected = expected_weighted(data, weights, dim, skipna, 'sum_of_weights')
    assert_allclose(expected, result)
    result = data.weighted(weights).sum(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, 'sum')
    assert_allclose(expected, result)
    result = data.weighted(weights).mean(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, 'mean')
    assert_allclose(expected, result)
    result = data.weighted(weights).sum_of_squares(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, 'sum_of_squares')
    assert_allclose(expected, result)
    result = data.weighted(weights).var(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, 'var')
    assert_allclose(expected, result)
    result = data.weighted(weights).std(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, 'std')
    assert_allclose(expected, result)