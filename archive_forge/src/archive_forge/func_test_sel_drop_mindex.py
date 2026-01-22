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
def test_sel_drop_mindex(self) -> None:
    midx = pd.MultiIndex.from_arrays([['a', 'a'], [1, 2]], names=('foo', 'bar'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    data = Dataset(coords=midx_coords)
    actual = data.sel(foo='a', drop=True)
    assert 'foo' not in actual.coords
    actual = data.sel(foo='a', drop=False)
    assert_equal(actual.foo, DataArray('a', coords={'foo': 'a'}))