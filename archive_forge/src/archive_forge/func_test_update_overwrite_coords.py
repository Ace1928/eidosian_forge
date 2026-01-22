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
def test_update_overwrite_coords(self) -> None:
    data = Dataset({'a': ('x', [1, 2])}, {'b': 3})
    data.update(Dataset(coords={'b': 4}))
    expected = Dataset({'a': ('x', [1, 2])}, {'b': 4})
    assert_identical(data, expected)
    data = Dataset({'a': ('x', [1, 2])}, {'b': 3})
    data.update(Dataset({'c': 5}, coords={'b': 4}))
    expected = Dataset({'a': ('x', [1, 2]), 'c': 5}, {'b': 4})
    assert_identical(data, expected)
    data = Dataset({'a': ('x', [1, 2])}, {'b': 3})
    data.update({'c': DataArray(5, coords={'b': 4})})
    expected = Dataset({'a': ('x', [1, 2]), 'c': 5}, {'b': 3})
    assert_identical(data, expected)