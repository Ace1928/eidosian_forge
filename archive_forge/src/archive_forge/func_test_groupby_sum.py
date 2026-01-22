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
def test_groupby_sum(self) -> None:
    array = self.da
    grouped = array.groupby('abc')
    expected_sum_all = Dataset({'foo': Variable(['abc'], np.array([self.x[:, :9].sum(), self.x[:, 10:].sum(), self.x[:, 9:10].sum()]).T), 'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
    assert_allclose(expected_sum_all, grouped.reduce(np.sum, dim=...))
    assert_allclose(expected_sum_all, grouped.sum(...))
    expected = DataArray([array['y'].values[idx].sum() for idx in [slice(9), slice(10, None), slice(9, 10)]], [['a', 'b', 'c']], ['abc'])
    actual = array['y'].groupby('abc').map(np.sum)
    assert_allclose(expected, actual)
    actual = array['y'].groupby('abc').sum(...)
    assert_allclose(expected, actual)
    expected_sum_axis1 = Dataset({'foo': (['x', 'abc'], np.array([self.x[:, :9].sum(1), self.x[:, 10:].sum(1), self.x[:, 9:10].sum(1)]).T), 'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
    assert_allclose(expected_sum_axis1, grouped.reduce(np.sum, 'y'))
    assert_allclose(expected_sum_axis1, grouped.sum('y'))