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
def test_setitem_dimension_override(self) -> None:
    ds = xr.Dataset({'x': [0, 1, 2]})
    ds['x'] = ds['x'][:2]
    expected = Dataset({'x': [0, 1]})
    assert_identical(ds, expected)
    ds = xr.Dataset({'x': [0, 1, 2]})
    ds['x'] = np.array([0, 1])
    assert_identical(ds, expected)
    ds = xr.Dataset({'x': [0, 1, 2]})
    ds.coords['x'] = [0, 1]
    assert_identical(ds, expected)