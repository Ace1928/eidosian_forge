from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('as_dataset', (True, False))
@pytest.mark.parametrize('weights', ([np.nan, 2], [np.nan, np.nan]))
def test_weighted_weights_nan_raises(as_dataset: bool, weights: list[float]) -> None:
    data: DataArray | Dataset = DataArray([1, 2])
    if as_dataset:
        data = data.to_dataset(name='data')
    with pytest.raises(ValueError, match='`weights` cannot contain missing values.'):
        data.weighted(DataArray(weights))