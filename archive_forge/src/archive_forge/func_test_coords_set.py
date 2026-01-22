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
def test_coords_set(self) -> None:
    one_coord = Dataset({'x': ('x', [0]), 'yy': ('x', [1]), 'zzz': ('x', [2])})
    two_coords = Dataset({'zzz': ('x', [2])}, {'x': ('x', [0]), 'yy': ('x', [1])})
    all_coords = Dataset(coords={'x': ('x', [0]), 'yy': ('x', [1]), 'zzz': ('x', [2])})
    actual = one_coord.set_coords('x')
    assert_identical(one_coord, actual)
    actual = one_coord.set_coords(['x'])
    assert_identical(one_coord, actual)
    actual = one_coord.set_coords('yy')
    assert_identical(two_coords, actual)
    actual = one_coord.set_coords(['yy', 'zzz'])
    assert_identical(all_coords, actual)
    actual = one_coord.reset_coords()
    assert_identical(one_coord, actual)
    actual = two_coords.reset_coords()
    assert_identical(one_coord, actual)
    actual = all_coords.reset_coords()
    assert_identical(one_coord, actual)
    actual = all_coords.reset_coords(['yy', 'zzz'])
    assert_identical(one_coord, actual)
    actual = all_coords.reset_coords('zzz')
    assert_identical(two_coords, actual)
    with pytest.raises(ValueError, match='cannot remove index'):
        one_coord.reset_coords('x')
    actual = all_coords.reset_coords('zzz', drop=True)
    expected = all_coords.drop_vars('zzz')
    assert_identical(expected, actual)
    expected = two_coords.drop_vars('zzz')
    assert_identical(expected, actual)