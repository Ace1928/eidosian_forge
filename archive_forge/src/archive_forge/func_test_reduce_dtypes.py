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
def test_reduce_dtypes(self) -> None:
    expected = Dataset({'x': 1})
    actual = Dataset({'x': True}).sum()
    assert_identical(expected, actual)
    expected = Dataset({'x': 3})
    actual = Dataset({'x': ('y', np.array([1, 2], 'uint16'))}).sum()
    assert_identical(expected, actual)
    expected = Dataset({'x': 1 + 1j})
    actual = Dataset({'x': ('y', [1, 1j])}).sum()
    assert_identical(expected, actual)