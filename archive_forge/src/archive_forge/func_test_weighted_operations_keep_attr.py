from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('operation', ('sum_of_weights', 'sum', 'mean', 'sum_of_squares', 'var', 'std', 'quantile'))
@pytest.mark.parametrize('as_dataset', (True, False))
@pytest.mark.parametrize('keep_attrs', (True, False, None))
def test_weighted_operations_keep_attr(operation, as_dataset, keep_attrs):
    weights = DataArray(np.random.randn(2, 2), attrs=dict(attr='weights'))
    data = DataArray(np.random.randn(2, 2))
    if as_dataset:
        data = data.to_dataset(name='data')
    data.attrs = dict(attr='weights')
    kwargs = {'keep_attrs': keep_attrs}
    if operation == 'quantile':
        kwargs['q'] = 0.5
    result = getattr(data.weighted(weights), operation)(**kwargs)
    if operation == 'sum_of_weights':
        assert result.attrs == (weights.attrs if keep_attrs else {})
        assert result.attrs == (weights.attrs if keep_attrs else {})
    else:
        assert result.attrs == (weights.attrs if keep_attrs else {})
        assert result.attrs == (data.attrs if keep_attrs else {})