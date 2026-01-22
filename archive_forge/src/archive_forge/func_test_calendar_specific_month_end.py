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
@pytest.mark.parametrize(('calendar', 'expected_month_day'), _CALENDAR_SPECIFIC_MONTH_END_TESTS, ids=_id_func)
def test_calendar_specific_month_end(calendar: str, expected_month_day: list[tuple[int, int]]) -> None:
    year = 2000
    result = cftime_range(start='2000-02', end='2001', freq='2ME', calendar=calendar).values
    date_type = get_date_type(calendar)
    expected = [date_type(year, *args) for args in expected_month_day]
    np.testing.assert_equal(result, expected)