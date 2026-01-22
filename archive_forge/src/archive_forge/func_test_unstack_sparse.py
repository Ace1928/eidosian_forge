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
@requires_sparse
def test_unstack_sparse(self) -> None:
    ds = xr.Dataset({'var': (('x',), np.arange(6))}, coords={'x': [0, 1, 2] * 2, 'y': (('x',), ['a'] * 3 + ['b'] * 3)})
    ds = ds.isel(x=[0, 2, 3, 4]).set_index(index=['x', 'y'])
    actual1 = ds.unstack('index', sparse=True)
    expected1 = ds.unstack('index')
    assert isinstance(actual1['var'].data, sparse_array_type)
    assert actual1['var'].variable._to_dense().equals(expected1['var'].variable)
    assert actual1['var'].data.density < 1.0
    actual2 = ds['var'].unstack('index', sparse=True)
    expected2 = ds['var'].unstack('index')
    assert isinstance(actual2.data, sparse_array_type)
    assert actual2.variable._to_dense().equals(expected2.variable)
    assert actual2.data.density < 1.0
    midx = pd.MultiIndex.from_arrays([np.arange(3), np.arange(3)], names=['a', 'b'])
    coords = Coordinates.from_pandas_multiindex(midx, 'z')
    coords['foo'] = np.arange(4)
    coords['bar'] = np.arange(5)
    ds_eye = Dataset({'var': (('z', 'foo', 'bar'), np.ones((3, 4, 5)))}, coords=coords)
    actual3 = ds_eye.unstack(sparse=True, fill_value=0)
    assert isinstance(actual3['var'].data, sparse_array_type)
    expected3 = xr.Dataset({'var': (('foo', 'bar', 'a', 'b'), np.broadcast_to(np.eye(3, 3), (4, 5, 3, 3)))}, coords={'foo': np.arange(4), 'bar': np.arange(5), 'a': np.arange(3), 'b': np.arange(3)})
    actual3['var'].data = actual3['var'].data.todense()
    assert_equal(expected3, actual3)