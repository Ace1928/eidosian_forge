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
def test_rename_multiindex(self) -> None:
    midx = pd.MultiIndex.from_tuples([[1, 2], [3, 4]], names=['a', 'b'])
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    original = Dataset({}, midx_coords)
    midx_renamed = midx.rename(['a', 'c'])
    midx_coords_renamed = Coordinates.from_pandas_multiindex(midx_renamed, 'x')
    expected = Dataset({}, midx_coords_renamed)
    actual = original.rename({'b': 'c'})
    assert_identical(expected, actual)
    with pytest.raises(ValueError, match="'a' conflicts"):
        with pytest.warns(UserWarning, match='does not create an index anymore'):
            original.rename({'x': 'a'})
    with pytest.raises(ValueError, match="'x' conflicts"):
        with pytest.warns(UserWarning, match='does not create an index anymore'):
            original.rename({'a': 'x'})
    with pytest.raises(ValueError, match="'b' conflicts"):
        original.rename({'a': 'b'})