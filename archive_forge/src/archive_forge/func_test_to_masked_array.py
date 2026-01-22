from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_to_masked_array(self) -> None:
    rs = np.random.RandomState(44)
    x = rs.random_sample(size=(10, 20))
    x_masked = np.ma.masked_where(x < 0.5, x)
    da = DataArray(x_masked)
    x_masked_2 = da.to_masked_array()
    da_2 = DataArray(x_masked_2)
    assert_array_equal(x_masked, x_masked_2)
    assert_equal(da, da_2)
    da_masked_array = da.to_masked_array(copy=True)
    assert isinstance(da_masked_array, np.ma.MaskedArray)
    assert_array_equal(da_masked_array.mask, x_masked.mask)
    assert_array_equal(da.values, x_masked.filled(np.nan))
    assert_array_equal(da_masked_array, x_masked.filled(np.nan))
    masked_array = da.to_masked_array(copy=False)
    masked_array[0, 0] = 10.0
    assert masked_array[0, 0] == 10.0
    assert da[0, 0].values == 10.0
    assert masked_array.base is da.values
    assert isinstance(masked_array, np.ma.MaskedArray)
    for v in [4, np.nan, True, '4', 'four']:
        da = DataArray(v)
        ma = da.to_masked_array()
        assert isinstance(ma, np.ma.MaskedArray)
    N = 4
    v = range(N)
    da = DataArray(v)
    ma = da.to_masked_array()
    assert len(ma.mask) == N