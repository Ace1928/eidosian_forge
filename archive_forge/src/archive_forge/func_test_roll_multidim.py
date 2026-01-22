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
def test_roll_multidim(self) -> None:
    arr = xr.DataArray([[1, 2, 3], [4, 5, 6]], coords={'x': range(3), 'y': range(2)}, dims=('y', 'x'))
    actual = arr.roll(x=1, roll_coords=True)
    expected = xr.DataArray([[3, 1, 2], [6, 4, 5]], coords=[('y', [0, 1]), ('x', [2, 0, 1])])
    assert_identical(expected, actual)