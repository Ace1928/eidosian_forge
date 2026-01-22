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
def test_reindex_regressions(self) -> None:
    da = DataArray(np.random.randn(5), coords=[('time', range(5))])
    time2 = DataArray(np.arange(5), dims='time2')
    with pytest.raises(ValueError):
        da.reindex(time=time2)
    xnp = np.array([1, 2, 3], dtype=complex)
    x = DataArray(xnp, coords=[[0.1, 0.2, 0.3]])
    y = DataArray([2, 5, 6, 7, 8], coords=[[-1.1, 0.21, 0.31, 0.41, 0.51]])
    re_dtype = x.reindex_like(y, method='pad').dtype
    assert x.dtype == re_dtype