from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_no_conflicts(self):
    ds1 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
    ds2 = xr.Dataset({'a': ('x', [2, 3]), 'x': [1, 2]})
    expected = xr.Dataset({'a': ('x', [1, 2, 3]), 'x': [0, 1, 2]})
    assert expected.identical(ds1.merge(ds2, compat='no_conflicts'))
    assert expected.identical(ds2.merge(ds1, compat='no_conflicts'))
    assert ds1.identical(ds1.merge(ds2, compat='no_conflicts', join='left'))
    assert ds2.identical(ds1.merge(ds2, compat='no_conflicts', join='right'))
    expected2 = xr.Dataset({'a': ('x', [2]), 'x': [1]})
    assert expected2.identical(ds1.merge(ds2, compat='no_conflicts', join='inner'))
    with pytest.raises(xr.MergeError):
        ds3 = xr.Dataset({'a': ('x', [99, 3]), 'x': [1, 2]})
        ds1.merge(ds3, compat='no_conflicts')
    with pytest.raises(xr.MergeError):
        ds3 = xr.Dataset({'a': ('y', [2, 3]), 'y': [1, 2]})
        ds1.merge(ds3, compat='no_conflicts')