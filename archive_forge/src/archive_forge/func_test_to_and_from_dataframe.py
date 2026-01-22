from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_to_and_from_dataframe(self) -> None:
    x = np.random.randn(10)
    y = np.random.randn(10)
    t = list('abcdefghij')
    ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
    expected = pd.DataFrame(np.array([x, y]).T, columns=['a', 'b'], index=pd.Index(t, name='t'))
    actual = ds.to_dataframe()
    assert expected.equals(actual), (expected, actual)
    actual = ds.set_coords('b').to_dataframe()
    assert expected.equals(actual), (expected, actual)
    assert_identical(ds, Dataset.from_dataframe(actual))
    w = np.random.randn(2, 3)
    ds = Dataset({'w': (('x', 'y'), w)})
    ds['y'] = ('y', list('abc'))
    exp_index = pd.MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']], names=['x', 'y'])
    expected = pd.DataFrame(w.reshape(-1), columns=['w'], index=exp_index)
    actual = ds.to_dataframe()
    assert expected.equals(actual)
    assert_identical(ds.assign_coords(x=[0, 1]), Dataset.from_dataframe(actual))
    new_order = ['x', 'y']
    actual = ds.to_dataframe(dim_order=new_order)
    assert expected.equals(actual)
    new_order = ['y', 'x']
    exp_index = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'c', 'c'], [0, 1, 0, 1, 0, 1]], names=['y', 'x'])
    expected = pd.DataFrame(w.transpose().reshape(-1), columns=['w'], index=exp_index)
    actual = ds.to_dataframe(dim_order=new_order)
    assert expected.equals(actual)
    invalid_order = ['x']
    with pytest.raises(ValueError, match='does not match the set of dimensions of this'):
        ds.to_dataframe(dim_order=invalid_order)
    invalid_order = ['x', 'z']
    with pytest.raises(ValueError, match='does not match the set of dimensions of this'):
        ds.to_dataframe(dim_order=invalid_order)
    df = pd.DataFrame([1])
    actual = Dataset.from_dataframe(df)
    expected = Dataset({0: ('index', [1])}, {'index': [0]})
    assert_identical(expected, actual)
    df = pd.DataFrame()
    actual = Dataset.from_dataframe(df)
    expected = Dataset(coords={'index': []})
    assert_identical(expected, actual)
    df = pd.DataFrame({'A': []})
    actual = Dataset.from_dataframe(df)
    expected = Dataset({'A': DataArray([], dims=('index',))}, {'index': []})
    assert_identical(expected, actual)
    ds = Dataset({'x': pd.Index(['bar']), 'a': ('y', np.array([1], 'int64'))}).isel(x=0)
    actual = ds.to_dataframe().loc[:, ['a', 'x']]
    expected = pd.DataFrame([[1, 'bar']], index=pd.Index([0], name='y'), columns=['a', 'x'])
    assert expected.equals(actual), (expected, actual)
    ds = Dataset({'x': np.array([0], 'int64'), 'y': np.array([1], 'int64')})
    actual = ds.to_dataframe()
    idx = pd.MultiIndex.from_arrays([[0], [1]], names=['x', 'y'])
    expected = pd.DataFrame([[]], index=idx)
    assert expected.equals(actual), (expected, actual)