from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_dataset_join() -> None:
    ds0 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
    ds1 = xr.Dataset({'a': ('x', [99, 3]), 'x': [1, 2]})
    with pytest.raises(ValueError, match='cannot align.*join.*exact.*'):
        apply_ufunc(operator.add, ds0, ds1)
    with pytest.raises(TypeError, match='must supply'):
        apply_ufunc(operator.add, ds0, ds1, dataset_join='outer')

    def add(a, b, join, dataset_join):
        return apply_ufunc(operator.add, a, b, join=join, dataset_join=dataset_join, dataset_fill_value=np.nan)
    actual = add(ds0, ds1, 'outer', 'inner')
    expected = xr.Dataset({'a': ('x', [np.nan, 101, np.nan]), 'x': [0, 1, 2]})
    assert_identical(actual, expected)
    actual = add(ds0, ds1, 'outer', 'outer')
    assert_identical(actual, expected)
    with pytest.raises(ValueError, match='data variable names'):
        apply_ufunc(operator.add, ds0, xr.Dataset({'b': 1}))
    ds2 = xr.Dataset({'b': ('x', [99, 3]), 'x': [1, 2]})
    actual = add(ds0, ds2, 'outer', 'inner')
    expected = xr.Dataset({'x': [0, 1, 2]})
    assert_identical(actual, expected)
    actual = add(ds0, ds2, 'outer', 'outer')
    expected = xr.Dataset({'a': ('x', [np.nan, np.nan, np.nan]), 'b': ('x', [np.nan, np.nan, np.nan]), 'x': [0, 1, 2]})
    assert_identical(actual, expected)