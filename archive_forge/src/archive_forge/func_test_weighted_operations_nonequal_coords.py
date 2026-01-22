from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('coords_weights, coords_data, expected_value_at_weighted_quantile', [([0, 1, 2, 3], [1, 2, 3, 4], 2.5), ([0, 1, 2, 3], [2, 3, 4, 5], 1.8), ([2, 3, 4, 5], [0, 1, 2, 3], 3.8)])
def test_weighted_operations_nonequal_coords(coords_weights: Iterable[Any], coords_data: Iterable[Any], expected_value_at_weighted_quantile: float) -> None:
    """Check that weighted operations work with unequal coords.


    Parameters
    ----------
    coords_weights : Iterable[Any]
        The coords for the weights.
    coords_data : Iterable[Any]
        The coords for the data.
    expected_value_at_weighted_quantile : float
        The expected value for the quantile of the weighted data.
    """
    da_weights = DataArray([0.5, 1.0, 1.0, 2.0], dims=('a',), coords=dict(a=coords_weights))
    da_data = DataArray([1, 2, 3, 4], dims=('a',), coords=dict(a=coords_data))
    check_weighted_operations(da_data, da_weights, dim='a', skipna=None)
    quantile = 0.5
    da_actual = da_data.weighted(da_weights).quantile(quantile, dim='a')
    da_expected = DataArray([expected_value_at_weighted_quantile], coords={'quantile': [quantile]}).squeeze()
    assert_allclose(da_actual, da_expected)
    ds_data = da_data.to_dataset(name='data')
    check_weighted_operations(ds_data, da_weights, dim='a', skipna=None)
    ds_actual = ds_data.weighted(da_weights).quantile(quantile, dim='a')
    assert_allclose(ds_actual, da_expected.to_dataset(name='data'))