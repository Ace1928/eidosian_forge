from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.parametrize(('start', 'end', 'periods', 'freq', 'inclusive', 'normalize', 'expected_date_args'), _CFTIME_RANGE_TESTS, ids=_id_func)
def test_cftime_range(start, end, periods, freq, inclusive, normalize, calendar, expected_date_args):
    date_type = get_date_type(calendar)
    expected_dates = [date_type(*args) for args in expected_date_args]
    if isinstance(start, tuple):
        start = date_type(*start)
    if isinstance(end, tuple):
        end = date_type(*end)
    result = cftime_range(start=start, end=end, periods=periods, freq=freq, inclusive=inclusive, normalize=normalize, calendar=calendar)
    resulting_dates = result.values
    assert isinstance(result, CFTimeIndex)
    if freq is not None:
        np.testing.assert_equal(resulting_dates, expected_dates)
    else:
        deltas = resulting_dates - expected_dates
        deltas = np.array([delta.total_seconds() for delta in deltas])
        assert np.max(np.abs(deltas)) < 0.001