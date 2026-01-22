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
def test_align_str_dtype(self) -> None:
    a = Dataset({'foo': ('x', [0, 1])}, coords={'x': ['a', 'b']})
    b = Dataset({'foo': ('x', [1, 2])}, coords={'x': ['b', 'c']})
    expected_a = Dataset({'foo': ('x', [0, 1, np.nan])}, coords={'x': ['a', 'b', 'c']})
    expected_b = Dataset({'foo': ('x', [np.nan, 1, 2])}, coords={'x': ['a', 'b', 'c']})
    actual_a, actual_b = xr.align(a, b, join='outer')
    assert_identical(expected_a, actual_a)
    assert expected_a.x.dtype == actual_a.x.dtype
    assert_identical(expected_b, actual_b)
    assert expected_b.x.dtype == actual_b.x.dtype