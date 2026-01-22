from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_attrs_override_copy(self):
    ds1 = xr.Dataset(attrs={'x': 0})
    ds2 = xr.Dataset(attrs={'x': 1})
    ds3 = xr.merge([ds1, ds2], combine_attrs='override')
    ds3.attrs['x'] = 2
    assert ds1.x == 0