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
def test_concat_join_kwarg(self) -> None:
    ds1 = Dataset({'a': (('x', 'y'), [[0]])}, coords={'x': [0], 'y': [0]}).to_dataarray()
    ds2 = Dataset({'a': (('x', 'y'), [[0]])}, coords={'x': [1], 'y': [0.0001]}).to_dataarray()
    expected: dict[JoinOptions, Any] = {}
    expected['outer'] = Dataset({'a': (('x', 'y'), [[0, np.nan], [np.nan, 0]])}, {'x': [0, 1], 'y': [0, 0.0001]})
    expected['inner'] = Dataset({'a': (('x', 'y'), [[], []])}, {'x': [0, 1], 'y': []})
    expected['left'] = Dataset({'a': (('x', 'y'), np.array([0, np.nan], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0]})
    expected['right'] = Dataset({'a': (('x', 'y'), np.array([np.nan, 0], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0.0001]})
    expected['override'] = Dataset({'a': (('x', 'y'), np.array([0, 0], ndmin=2).T)}, coords={'x': [0, 1], 'y': [0]})
    with pytest.raises(ValueError, match="cannot align.*exact.*dimensions.*'y'"):
        actual = concat([ds1, ds2], join='exact', dim='x')
    for join in expected:
        actual = concat([ds1, ds2], join=join, dim='x')
        assert_equal(actual, expected[join].to_dataarray())