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
def test_groupby_dataset() -> None:
    data = Dataset({'z': (['x', 'y'], np.random.randn(3, 5))}, {'x': ('x', list('abc')), 'c': ('x', [0, 1, 0]), 'y': range(5)})
    groupby = data.groupby('x', squeeze=False)
    assert len(groupby) == 3
    expected_groups = {'a': slice(0, 1), 'b': slice(1, 2), 'c': slice(2, 3)}
    assert groupby.groups == expected_groups
    expected_items = [('a', data.isel(x=[0])), ('b', data.isel(x=[1])), ('c', data.isel(x=[2]))]
    for actual1, expected1 in zip(groupby, expected_items):
        assert actual1[0] == expected1[0]
        assert_equal(actual1[1], expected1[1])

    def identity(x):
        return x
    for k in ['x', 'c', 'y']:
        actual2 = data.groupby(k, squeeze=False).map(identity)
        assert_equal(data, actual2)