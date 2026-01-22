from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@requires_dask
@pytest.mark.parametrize('as_dataset', (True, False))
@pytest.mark.parametrize('weights', ([np.nan, 2], [np.nan, np.nan]))
def test_weighted_weights_nan_raises_dask(as_dataset, weights):
    data = DataArray([1, 2]).chunk({'dim_0': -1})
    if as_dataset:
        data = data.to_dataset(name='data')
    weights = DataArray(weights).chunk({'dim_0': -1})
    with raise_if_dask_computes():
        weighted = data.weighted(weights)
    with pytest.raises(ValueError, match='`weights` cannot contain missing values.'):
        weighted.sum().load()