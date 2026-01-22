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
def test_update_auto_align(self) -> None:
    ds = Dataset({'x': ('t', [3, 4])}, {'t': [0, 1]})
    expected1 = Dataset({'x': ('t', [3, 4]), 'y': ('t', [np.nan, 5])}, {'t': [0, 1]})
    actual1 = ds.copy()
    other1 = {'y': ('t', [5]), 't': [1]}
    with pytest.raises(ValueError, match='conflicting sizes'):
        actual1.update(other1)
    actual1.update(Dataset(other1))
    assert_identical(expected1, actual1)
    actual2 = ds.copy()
    other2 = Dataset({'y': ('t', [5]), 't': [100]})
    actual2.update(other2)
    expected2 = Dataset({'x': ('t', [3, 4]), 'y': ('t', [np.nan] * 2)}, {'t': [0, 1]})
    assert_identical(expected2, actual2)