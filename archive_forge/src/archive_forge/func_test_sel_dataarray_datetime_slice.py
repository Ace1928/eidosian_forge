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
def test_sel_dataarray_datetime_slice(self) -> None:
    times = pd.date_range('2000-01-01', freq='D', periods=365)
    array = DataArray(np.arange(365), [('time', times)])
    result = array.sel(time=slice(array.time[0], array.time[-1]))
    assert_equal(result, array)
    array = DataArray(np.arange(365), [('delta', times - times[0])])
    result = array.sel(delta=slice(array.delta[0], array.delta[-1]))
    assert_equal(result, array)