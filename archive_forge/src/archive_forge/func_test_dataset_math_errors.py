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
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
def test_dataset_math_errors(self) -> None:
    ds = self.make_example_math_dataset()
    with pytest.raises(TypeError):
        ds['foo'] += ds
    with pytest.raises(TypeError):
        ds['foo'].variable += ds
    with pytest.raises(ValueError, match='must have the same'):
        ds += ds[['bar']]
    other = DataArray(np.datetime64('2000-01-01'), coords={'c': 2})
    actual = ds.copy(deep=True)
    with pytest.raises(TypeError):
        actual += other
    assert_identical(actual, ds)