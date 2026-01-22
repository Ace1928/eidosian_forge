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
@pytest.mark.parametrize(['arg', 'drop', 'dropped', 'converted', 'renamed'], [('foo', False, [], [], {'bar': 'x'}), ('foo', True, ['foo'], [], {'bar': 'x'}), ('x', False, ['x'], ['foo', 'bar'], {}), ('x', True, ['x', 'foo', 'bar'], [], {}), (['foo', 'bar'], False, ['x'], ['foo', 'bar'], {}), (['foo', 'bar'], True, ['x', 'foo', 'bar'], [], {}), (['x', 'foo'], False, ['x'], ['foo', 'bar'], {}), (['foo', 'x'], True, ['x', 'foo', 'bar'], [], {})])
def test_reset_index_drop_convert(self, arg: str | list[str], drop: bool, dropped: list[str], converted: list[str], renamed: dict[str, str]) -> None:
    midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=('foo', 'bar'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    ds = xr.Dataset(coords=midx_coords)
    reset = ds.reset_index(arg, drop=drop)
    for name in dropped:
        assert name not in reset.variables
    for name in converted:
        assert_identical(reset[name].variable, ds[name].variable.to_base_variable())
    for old_name, new_name in renamed.items():
        assert_identical(ds[old_name].variable, reset[new_name].variable)