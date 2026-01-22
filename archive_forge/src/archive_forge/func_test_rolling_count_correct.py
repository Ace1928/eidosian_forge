from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
def test_rolling_count_correct(self, compute_backend) -> None:
    da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims='time')
    kwargs: list[dict[str, Any]] = [{'time': 11, 'min_periods': 1}, {'time': 11, 'min_periods': None}, {'time': 7, 'min_periods': 2}]
    expecteds = [DataArray([1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8], dims='time'), DataArray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dims='time'), DataArray([np.nan, np.nan, 2, 3, 3, 4, 5, 5, 5, 5, 5], dims='time')]
    for kwarg, expected in zip(kwargs, expecteds):
        result = da.rolling(**kwarg).count()
        assert_equal(result, expected)
        result = da.to_dataset(name='var1').rolling(**kwarg).count()['var1']
        assert_equal(result, expected)