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
def test_getitem_coords(self) -> None:
    orig = DataArray([[10], [20]], {'x': [1, 2], 'y': [3], 'z': 4, 'x2': ('x', ['a', 'b']), 'y2': ('y', ['c']), 'xy': (['y', 'x'], [['d', 'e']])}, dims=['x', 'y'])
    assert_identical(orig, orig[:])
    assert_identical(orig, orig[:, :])
    assert_identical(orig, orig[...])
    assert_identical(orig, orig[:2, :1])
    assert_identical(orig, orig[[0, 1], [0]])
    actual = orig[0, 0]
    expected = DataArray(10, {'x': 1, 'y': 3, 'z': 4, 'x2': 'a', 'y2': 'c', 'xy': 'd'})
    assert_identical(expected, actual)
    actual = orig[0, :]
    expected = DataArray([10], {'x': 1, 'y': [3], 'z': 4, 'x2': 'a', 'y2': ('y', ['c']), 'xy': ('y', ['d'])}, dims='y')
    assert_identical(expected, actual)
    actual = orig[:, 0]
    expected = DataArray([10, 20], {'x': [1, 2], 'y': 3, 'z': 4, 'x2': ('x', ['a', 'b']), 'y2': 'c', 'xy': ('x', ['d', 'e'])}, dims='x')
    assert_identical(expected, actual)