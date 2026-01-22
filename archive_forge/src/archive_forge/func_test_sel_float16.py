from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_sel_float16(self) -> None:
    data_values = np.arange(4)
    coord_values = np.array([0.0, 0.111, 0.222, 0.333], dtype='float16')
    indices = slice(1, 3)
    message = '`pandas.Index` does not support the `float16` dtype.*'
    with pytest.warns(DeprecationWarning, match=message):
        arr = DataArray(data_values, coords={'x': coord_values}, dims='x')
    with pytest.warns(DeprecationWarning, match=message):
        expected = DataArray(data_values[indices], coords={'x': coord_values[indices]}, dims='x')
    actual = arr.sel(x=coord_values[indices])
    assert_equal(actual, expected)