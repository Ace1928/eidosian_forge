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
def test_drop_index_labels(self) -> None:
    data = Dataset({'A': (['x', 'y'], np.random.randn(2, 3)), 'x': ['a', 'b']})
    with pytest.warns(DeprecationWarning):
        actual = data.drop(['a'], dim='x')
    expected = data.isel(x=[1])
    assert_identical(expected, actual)
    with pytest.warns(DeprecationWarning):
        actual = data.drop(['a', 'b'], dim='x')
    expected = data.isel(x=slice(0, 0))
    assert_identical(expected, actual)
    with pytest.raises(KeyError):
        with pytest.warns(DeprecationWarning):
            data.drop(['c'], dim='x')
    with pytest.warns(DeprecationWarning):
        actual = data.drop(['c'], dim='x', errors='ignore')
    assert_identical(data, actual)
    with pytest.raises(ValueError):
        data.drop(['c'], dim='x', errors='wrong_value')
    with pytest.warns(DeprecationWarning):
        actual = data.drop(['a', 'b', 'c'], 'x', errors='ignore')
    expected = data.isel(x=slice(0, 0))
    assert_identical(expected, actual)
    actual = data.drop_sel(x=DataArray(['a', 'b', 'c']), errors='ignore')
    expected = data.isel(x=slice(0, 0))
    assert_identical(expected, actual)
    with pytest.warns(DeprecationWarning):
        data.drop(DataArray(['a', 'b', 'c']), dim='x', errors='ignore')
    assert_identical(expected, actual)
    actual = data.drop_sel(y=[1])
    expected = data.isel(y=[0, 2])
    assert_identical(expected, actual)
    with pytest.raises(KeyError, match='not found in axis'):
        data.drop_sel(x=0)