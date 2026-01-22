from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@requires_scipy
def test_datetime_interp_noerror() -> None:
    a = xr.DataArray(np.arange(21).reshape(3, 7), dims=['x', 'time'], coords={'x': [1, 2, 3], 'time': pd.date_range('01-01-2001', periods=7, freq='D')})
    xi = xr.DataArray(np.linspace(1, 3, 50), dims=['time'], coords={'time': pd.date_range('01-01-2001', periods=50, freq='h')})
    a.interp(x=xi, time=xi.time)