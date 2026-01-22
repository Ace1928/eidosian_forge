from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('dim', ('a', 'b', 'c', ('a', 'b'), ('a', 'b', 'c'), None))
@pytest.mark.parametrize('add_nans', (True, False))
@pytest.mark.parametrize('skipna', (None, True, False))
@pytest.mark.filterwarnings('ignore:invalid value encountered in sqrt')
def test_weighted_operations_3D(dim, add_nans, skipna):
    dims = ('a', 'b', 'c')
    coords = dict(a=[0, 1, 2, 3], b=[0, 1, 2, 3], c=[0, 1, 2, 3])
    weights = DataArray(np.random.randn(4, 4, 4), dims=dims, coords=coords)
    data = np.random.randn(4, 4, 4)
    if add_nans:
        c = int(data.size * 0.25)
        data.ravel()[np.random.choice(data.size, c, replace=False)] = np.nan
    data = DataArray(data, dims=dims, coords=coords)
    check_weighted_operations(data, weights, dim, skipna)
    data = data.to_dataset(name='data')
    check_weighted_operations(data, weights, dim, skipna)