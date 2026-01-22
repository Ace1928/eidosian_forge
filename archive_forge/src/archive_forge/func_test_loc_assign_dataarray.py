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
def test_loc_assign_dataarray(self) -> None:

    def get_data():
        return DataArray(np.ones((4, 3, 2)), dims=['x', 'y', 'z'], coords={'x': np.arange(4), 'y': ['a', 'b', 'c'], 'non-dim': ('x', [1, 3, 4, 2])})
    da = get_data()
    ind = DataArray(np.arange(1, 4), dims=['y'], coords={'y': np.random.randn(3)})
    with pytest.raises(IndexError, match="dimension coordinate 'y'"):
        da.loc[dict(x=ind)] = 0
    ind = DataArray(np.arange(1, 4), dims=['x'], coords={'x': np.arange(1, 4)})
    da.loc[dict(x=ind)] = 0
    assert np.allclose(da[dict(x=ind)].values, 0)
    assert_identical(da['x'], get_data()['x'])
    assert_identical(da['non-dim'], get_data()['non-dim'])
    da = get_data()
    value = xr.DataArray(np.zeros((3, 3, 2)), dims=['x', 'y', 'z'], coords={'x': [0, 1, 2], 'non-dim': ('x', [0, 2, 4])})
    with pytest.raises(IndexError, match="dimension coordinate 'x'"):
        da.loc[dict(x=ind)] = value
    value = xr.DataArray(np.zeros((3, 3, 2)), dims=['x', 'y', 'z'], coords={'x': [1, 2, 3], 'non-dim': ('x', [0, 2, 4])})
    da.loc[dict(x=ind)] = value
    assert np.allclose(da[dict(x=ind)].values, 0)
    assert_identical(da['x'], get_data()['x'])
    assert_identical(da['non-dim'], get_data()['non-dim'])