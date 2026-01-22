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
@pytest.mark.parametrize('fill_value', [dtypes.NA, 2, 2.0, {'x': 2, 'z': 1}])
def test_reindex_like_fill_value(self, fill_value) -> None:
    ds = Dataset({'x': ('y', [10, 20]), 'z': ('y', [-20, -10]), 'y': [0, 1]})
    y = [0, 1, 2]
    alt = Dataset({'y': y})
    actual = ds.reindex_like(alt, fill_value=fill_value)
    if fill_value == dtypes.NA:
        fill_value_x = fill_value_z = np.nan
    elif isinstance(fill_value, dict):
        fill_value_x = fill_value['x']
        fill_value_z = fill_value['z']
    else:
        fill_value_x = fill_value_z = fill_value
    expected = Dataset({'x': ('y', [10, 20, fill_value_x]), 'z': ('y', [-20, -10, fill_value_z]), 'y': y})
    assert_identical(expected, actual)