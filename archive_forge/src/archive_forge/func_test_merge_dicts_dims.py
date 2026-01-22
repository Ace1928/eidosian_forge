from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_dicts_dims(self):
    actual = xr.merge([{'y': ('x', [13])}, {'x': [12]}])
    expected = xr.Dataset({'x': [12], 'y': ('x', [13])})
    assert_identical(actual, expected)