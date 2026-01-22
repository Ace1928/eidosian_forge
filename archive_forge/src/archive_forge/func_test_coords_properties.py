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
def test_coords_properties(self) -> None:
    data = Dataset({'x': ('x', np.array([-1, -2], 'int64')), 'y': ('y', np.array([0, 1, 2], 'int64')), 'foo': (['x', 'y'], np.random.randn(2, 3))}, {'a': ('x', np.array([4, 5], 'int64')), 'b': np.int64(-10)})
    coords = data.coords
    assert isinstance(coords, DatasetCoordinates)
    assert len(coords) == 4
    assert list(coords) == ['x', 'y', 'a', 'b']
    assert_identical(coords['x'].variable, data['x'].variable)
    assert_identical(coords['y'].variable, data['y'].variable)
    assert 'x' in coords
    assert 'a' in coords
    assert 0 not in coords
    assert 'foo' not in coords
    with pytest.raises(KeyError):
        coords['foo']
    with pytest.raises(KeyError):
        coords[0]
    expected = dedent('        Coordinates:\n          * x        (x) int64 16B -1 -2\n          * y        (y) int64 24B 0 1 2\n            a        (x) int64 16B 4 5\n            b        int64 8B -10')
    actual = repr(coords)
    assert expected == actual
    assert coords.sizes == {'x': 2, 'y': 3}
    assert coords.dtypes == {'x': np.dtype('int64'), 'y': np.dtype('int64'), 'a': np.dtype('int64'), 'b': np.dtype('int64')}