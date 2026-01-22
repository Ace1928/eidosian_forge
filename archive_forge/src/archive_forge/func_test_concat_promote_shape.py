from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def test_concat_promote_shape(self) -> None:
    objs = [Dataset({}, {'x': 0}), Dataset({'x': [1]})]
    actual = concat(objs, 'x')
    expected = Dataset({'x': [0, 1]})
    assert_identical(actual, expected)
    objs = [Dataset({'x': [0]}), Dataset({}, {'x': 1})]
    actual = concat(objs, 'x')
    assert_identical(actual, expected)
    objs = [Dataset({'x': [2], 'y': 3}), Dataset({'x': [4], 'y': 5})]
    actual = concat(objs, 'x')
    expected = Dataset({'x': [2, 4], 'y': ('x', [3, 5])})
    assert_identical(actual, expected)
    objs = [Dataset({'x': [0]}, {'y': -1}), Dataset({'x': [1]}, {'y': ('x', [-2])})]
    actual = concat(objs, 'x')
    expected = Dataset({'x': [0, 1]}, {'y': ('x', [-1, -2])})
    assert_identical(actual, expected)
    objs = [Dataset({'x': [0]}, {'y': -1}), Dataset({'x': [1, 2]}, {'y': -2})]
    actual = concat(objs, 'x')
    expected = Dataset({'x': [0, 1, 2]}, {'y': ('x', [-1, -2, -2])})
    assert_identical(actual, expected)
    objs = [Dataset({'z': ('x', [-1])}, {'x': [0], 'y': [0]}), Dataset({'z': ('y', [1])}, {'x': [1], 'y': [0]})]
    actual = concat(objs, 'x')
    expected = Dataset({'z': (('x', 'y'), [[-1], [1]])}, {'x': [0, 1], 'y': [0]})
    assert_identical(actual, expected)
    objs = [Dataset({}, {'x': pd.Interval(-1, 0, closed='right')}), Dataset({'x': [pd.Interval(0, 1, closed='right')]})]
    actual = concat(objs, 'x')
    expected = Dataset({'x': [pd.Interval(-1, 0, closed='right'), pd.Interval(0, 1, closed='right')]})
    assert_identical(actual, expected)
    time_data1 = np.array(['2022-01-01', '2022-02-01'], dtype='datetime64[ns]')
    time_data2 = np.array('2022-03-01', dtype='datetime64[ns]')
    time_expected = np.array(['2022-01-01', '2022-02-01', '2022-03-01'], dtype='datetime64[ns]')
    objs = [Dataset({}, {'time': time_data1}), Dataset({}, {'time': time_data2})]
    actual = concat(objs, 'time')
    expected = Dataset({}, {'time': time_expected})
    assert_identical(actual, expected)
    assert isinstance(actual.indexes['time'], pd.DatetimeIndex)