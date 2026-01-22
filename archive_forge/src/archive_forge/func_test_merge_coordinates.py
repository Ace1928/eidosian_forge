from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_coordinates(self):
    coords1 = xr.Coordinates({'x': ('x', [0, 1, 2])})
    coords2 = xr.Coordinates({'y': ('y', [3, 4, 5])})
    expected = xr.Dataset(coords={'x': [0, 1, 2], 'y': [3, 4, 5]})
    actual = xr.merge([coords1, coords2])
    assert_identical(actual, expected)