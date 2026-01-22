from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('operation', ('sum_of_weights', 'sum', 'mean', 'sum_of_squares', 'var', 'std', 'quantile'))
def test_weighted_operations_keep_attr_da_in_ds(operation):
    weights = DataArray(np.random.randn(2, 2))
    data = DataArray(np.random.randn(2, 2), attrs=dict(attr='data'))
    data = data.to_dataset(name='a')
    kwargs = {'keep_attrs': True}
    if operation == 'quantile':
        kwargs['q'] = 0.5
    result = getattr(data.weighted(weights), operation)(**kwargs)
    assert data.a.attrs == result.a.attrs