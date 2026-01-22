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
def test_binary_op_join_setting(self) -> None:
    missing_2 = xr.Dataset({'x': [0, 1]})
    missing_0 = xr.Dataset({'x': [1, 2]})
    with xr.set_options(arithmetic_join='outer'):
        actual = missing_2 + missing_0
    expected = xr.Dataset({'x': [0, 1, 2]})
    assert_equal(actual, expected)
    ds1 = xr.Dataset({'foo': 1, 'bar': 2})
    ds2 = xr.Dataset({'bar': 2, 'baz': 3})
    expected = xr.Dataset({'bar': 4})
    actual = ds1 + ds2
    assert_equal(actual, expected)
    with xr.set_options(arithmetic_join='outer'):
        expected = xr.Dataset({'foo': np.nan, 'bar': 4, 'baz': np.nan})
        actual = ds1 + ds2
        assert_equal(actual, expected)
    with xr.set_options(arithmetic_join='left'):
        expected = xr.Dataset({'foo': np.nan, 'bar': 4})
        actual = ds1 + ds2
        assert_equal(actual, expected)
    with xr.set_options(arithmetic_join='right'):
        expected = xr.Dataset({'bar': 4, 'baz': np.nan})
        actual = ds1 + ds2
        assert_equal(actual, expected)