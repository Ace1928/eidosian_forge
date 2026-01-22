from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_dicts_simple(self):
    actual = xr.merge([{'foo': 0}, {'bar': 'one'}, {'baz': 3.5}])
    expected = xr.Dataset({'foo': 0, 'bar': 'one', 'baz': 3.5})
    assert_identical(actual, expected)