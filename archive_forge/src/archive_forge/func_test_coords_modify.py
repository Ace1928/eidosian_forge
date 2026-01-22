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
def test_coords_modify(self) -> None:
    data = Dataset({'x': ('x', [-1, -2]), 'y': ('y', [0, 1, 2]), 'foo': (['x', 'y'], np.random.randn(2, 3))}, {'a': ('x', [4, 5]), 'b': -10})
    actual = data.copy(deep=True)
    actual.coords['x'] = ('x', ['a', 'b'])
    assert_array_equal(actual['x'], ['a', 'b'])
    actual = data.copy(deep=True)
    actual.coords['z'] = ('z', ['a', 'b'])
    assert_array_equal(actual['z'], ['a', 'b'])
    actual = data.copy(deep=True)
    with pytest.raises(ValueError, match='conflicting dimension sizes'):
        actual.coords['x'] = ('x', [-1])
    assert_identical(actual, data)
    actual = data.copy()
    del actual.coords['b']
    expected = data.reset_coords('b', drop=True)
    assert_identical(expected, actual)
    with pytest.raises(KeyError):
        del data.coords['not_found']
    with pytest.raises(KeyError):
        del data.coords['foo']
    actual = data.copy(deep=True)
    actual.coords.update({'c': 11})
    expected = data.merge({'c': 11}).set_coords('c')
    assert_identical(expected, actual)
    del actual.coords['x']
    assert 'x' not in actual.xindexes