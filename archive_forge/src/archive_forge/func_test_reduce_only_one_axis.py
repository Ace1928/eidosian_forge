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
def test_reduce_only_one_axis(self) -> None:

    def mean_only_one_axis(x, axis):
        if not isinstance(axis, integer_types):
            raise TypeError('non-integer axis')
        return x.mean(axis)
    ds = Dataset({'a': (['x', 'y'], [[0, 1, 2, 3, 4]])})
    expected = Dataset({'a': ('x', [2])})
    actual = ds.reduce(mean_only_one_axis, 'y')
    assert_identical(expected, actual)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'axis'"):
        ds.reduce(mean_only_one_axis)