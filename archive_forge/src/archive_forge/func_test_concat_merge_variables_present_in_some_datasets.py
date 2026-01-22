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
def test_concat_merge_variables_present_in_some_datasets(self, data) -> None:
    ds1 = Dataset(data_vars={'a': ('y', [0.1])}, coords={'x': 0.1})
    ds2 = Dataset(data_vars={'a': ('y', [0.2])}, coords={'z': 0.2})
    actual = concat([ds1, ds2], dim='y', coords='minimal')
    expected = Dataset({'a': ('y', [0.1, 0.2])}, coords={'x': 0.1, 'z': 0.2})
    assert_identical(expected, actual)
    split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]
    data0, data1 = deepcopy(split_data)
    data1['foo'] = ('bar', np.random.randn(10))
    actual = concat([data0, data1], 'dim1', data_vars='minimal')
    expected = data.copy().assign(foo=data1.foo)
    assert_identical(expected, actual)
    actual = concat([data0, data1], 'dim1')
    foo = np.ones((8, 10), dtype=data1.foo.dtype) * np.nan
    foo[3:] = data1.foo.values[None, ...]
    expected = data.copy().assign(foo=(['dim1', 'bar'], foo))
    assert_identical(expected, actual)