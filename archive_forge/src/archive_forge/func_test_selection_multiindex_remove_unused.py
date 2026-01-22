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
def test_selection_multiindex_remove_unused(self) -> None:
    ds = xr.DataArray(np.arange(40).reshape(8, 5), dims=['x', 'y'], coords={'x': np.arange(8), 'y': np.arange(5)})
    ds = ds.stack(xy=['x', 'y'])
    ds_isel = ds.isel(xy=ds['x'] < 4)
    with pytest.raises(KeyError):
        ds_isel.sel(x=5)
    actual = ds_isel.unstack()
    expected = ds.reset_index('xy').isel(xy=ds['x'] < 4)
    expected = expected.set_index(xy=['x', 'y']).unstack()
    assert_identical(expected, actual)