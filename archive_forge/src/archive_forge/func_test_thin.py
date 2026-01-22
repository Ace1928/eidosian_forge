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
def test_thin(self) -> None:
    data = create_test_data()
    expected = data.isel(time=slice(None, None, 5), dim2=slice(None, None, 6))
    actual = data.thin(time=5, dim2=6)
    assert_equal(expected, actual)
    expected = data.isel({dim: slice(None, None, 6) for dim in data.dims})
    actual = data.thin(6)
    assert_equal(expected, actual)
    with pytest.raises(TypeError, match='either dict-like or a single int'):
        data.thin([3])
    with pytest.raises(TypeError, match='expected integer type'):
        data.thin(dim2=3.1)
    with pytest.raises(ValueError, match='cannot be zero'):
        data.thin(time=0)
    with pytest.raises(ValueError, match='expected positive int'):
        data.thin(time=-3)