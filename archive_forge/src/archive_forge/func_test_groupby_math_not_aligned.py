from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_groupby_math_not_aligned(self) -> None:
    array = DataArray(range(4), {'b': ('x', [0, 0, 1, 1]), 'x': [0, 1, 2, 3]}, dims='x')
    other = DataArray([10], coords={'b': [0]}, dims='b')
    actual = array.groupby('b') + other
    expected = DataArray([10, 11, np.nan, np.nan], array.coords)
    assert_identical(expected, actual)
    other = array.groupby('b').sum()
    actual = array.sel(x=[0, 1]).groupby('b') - other
    expected = DataArray([-1, 0], {'b': ('x', [0, 0]), 'x': [0, 1]}, dims='x')
    assert_identical(expected, actual)
    other = DataArray([10], coords={'c': 123, 'b': [0]}, dims='b')
    actual = array.groupby('b') + other
    expected = DataArray([10, 11, np.nan, np.nan], array.coords)
    expected.coords['c'] = (['x'], [123] * 2 + [np.nan] * 2)
    assert_identical(expected, actual)
    other_ds = Dataset({'a': ('b', [10])}, {'b': [0]})
    actual_ds = array.groupby('b') + other_ds
    expected_ds = Dataset({'a': ('x', [10, 11, np.nan, np.nan])}, array.coords)
    assert_identical(expected_ds, actual_ds)