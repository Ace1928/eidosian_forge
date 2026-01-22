from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0, {'a': 2, 'b': 1}])
def test_merge_fill_value(self, fill_value):
    ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
    ds2 = xr.Dataset({'b': ('x', [3, 4]), 'x': [1, 2]})
    if fill_value == dtypes.NA:
        fill_value_a = fill_value_b = np.nan
    elif isinstance(fill_value, dict):
        fill_value_a = fill_value['a']
        fill_value_b = fill_value['b']
    else:
        fill_value_a = fill_value_b = fill_value
    expected = xr.Dataset({'a': ('x', [1, 2, fill_value_a]), 'b': ('x', [fill_value_b, 3, 4])}, {'x': [0, 1, 2]})
    assert expected.identical(ds1.merge(ds2, fill_value=fill_value))
    assert expected.identical(ds2.merge(ds1, fill_value=fill_value))
    assert expected.identical(xr.merge([ds1, ds2], fill_value=fill_value))