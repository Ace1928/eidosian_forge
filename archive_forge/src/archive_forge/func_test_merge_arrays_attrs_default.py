from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_arrays_attrs_default(self):
    var1_attrs = {'a': 1, 'b': 2}
    var2_attrs = {'a': 1, 'c': 3}
    expected_attrs = {'a': 1, 'b': 2}
    data = create_test_data(add_attrs=False)
    expected = data[['var1', 'var2']].copy()
    expected.var1.attrs = var1_attrs
    expected.var2.attrs = var2_attrs
    expected.attrs = expected_attrs
    data.var1.attrs = var1_attrs
    data.var2.attrs = var2_attrs
    actual = xr.merge([data.var1, data.var2])
    assert_identical(actual, expected)