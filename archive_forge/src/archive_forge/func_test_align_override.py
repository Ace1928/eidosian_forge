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
def test_align_override(self) -> None:
    left = xr.Dataset(coords={'x': [0, 1, 2]})
    right = xr.Dataset(coords={'x': [0.1, 1.1, 2.1], 'y': [1, 2, 3]})
    expected_right = xr.Dataset(coords={'x': [0, 1, 2], 'y': [1, 2, 3]})
    new_left, new_right = xr.align(left, right, join='override')
    assert_identical(left, new_left)
    assert_identical(new_right, expected_right)
    new_left, new_right = xr.align(left, right, exclude='x', join='override')
    assert_identical(left, new_left)
    assert_identical(right, new_right)
    new_left, new_right = xr.align(left.isel(x=0, drop=True), right, exclude='x', join='override')
    assert_identical(left.isel(x=0, drop=True), new_left)
    assert_identical(right, new_right)
    with pytest.raises(ValueError, match='cannot align.*join.*override.*same size'):
        xr.align(left.isel(x=0).expand_dims('x'), right, join='override')