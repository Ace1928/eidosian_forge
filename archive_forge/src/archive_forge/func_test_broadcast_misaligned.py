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
def test_broadcast_misaligned(self) -> None:
    x = Dataset({'foo': DataArray([1, 2, 3], coords=[('x', [-1, -2, -3])])})
    y = Dataset({'bar': DataArray([[1, 2], [3, 4]], dims=['y', 'x'], coords={'y': [1, 2], 'x': [10, -3]})})
    x2, y2 = broadcast(x, y)
    expected_x2 = Dataset({'foo': DataArray([[3, 3], [2, 2], [1, 1], [np.nan, np.nan]], dims=['x', 'y'], coords={'y': [1, 2], 'x': [-3, -2, -1, 10]})})
    expected_y2 = Dataset({'bar': DataArray([[2, 4], [np.nan, np.nan], [np.nan, np.nan], [1, 3]], dims=['x', 'y'], coords={'y': [1, 2], 'x': [-3, -2, -1, 10]})})
    assert_identical(expected_x2, x2)
    assert_identical(expected_y2, y2)