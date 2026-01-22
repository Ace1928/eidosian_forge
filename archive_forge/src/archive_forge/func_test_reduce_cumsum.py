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
def test_reduce_cumsum(self) -> None:
    data = xr.Dataset({'a': 1, 'b': ('x', [1, 2]), 'c': (('x', 'y'), [[np.nan, 3], [0, 4]])})
    assert_identical(data.fillna(0), data.cumsum('y'))
    expected = xr.Dataset({'a': 1, 'b': ('x', [1, 3]), 'c': (('x', 'y'), [[0, 3], [0, 7]])})
    assert_identical(expected, data.cumsum())