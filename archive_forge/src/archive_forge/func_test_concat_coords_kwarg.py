from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('coords', ['different', 'minimal', 'all'])
@pytest.mark.parametrize('dim', ['dim1', 'dim2'])
def test_concat_coords_kwarg(self, data, dim, coords) -> None:
    data = data.copy(deep=True)
    data.coords['extra'] = ('dim4', np.arange(3))
    datasets = [g.squeeze() for _, g in data.groupby(dim, squeeze=False)]
    actual = concat(datasets, data[dim], coords=coords)
    if coords == 'all':
        expected = np.array([data['extra'].values for _ in range(data.sizes[dim])])
        assert_array_equal(actual['extra'].values, expected)
    else:
        assert_equal(data['extra'], actual['extra'])