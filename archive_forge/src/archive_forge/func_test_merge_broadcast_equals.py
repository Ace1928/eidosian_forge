from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_broadcast_equals(self):
    ds1 = xr.Dataset({'x': 0})
    ds2 = xr.Dataset({'x': ('y', [0, 0])})
    actual = ds1.merge(ds2)
    assert_identical(ds2, actual)
    actual = ds2.merge(ds1)
    assert_identical(ds2, actual)
    actual = ds1.copy()
    actual.update(ds2)
    assert_identical(ds2, actual)
    ds1 = xr.Dataset({'x': np.nan})
    ds2 = xr.Dataset({'x': ('y', [np.nan, np.nan])})
    actual = ds1.merge(ds2)
    assert_identical(ds2, actual)