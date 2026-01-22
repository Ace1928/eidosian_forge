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
@pytest.mark.parametrize(('date_args', 'offset', 'expected'), [((1, 1, 1), MonthBegin(), True), ((1, 1, 1, 1), MonthBegin(), True), ((1, 1, 5), MonthBegin(), False), ((1, 1, 5), MonthEnd(), False), ((1, 3, 1), QuarterBegin(), True), ((1, 3, 1, 1), QuarterBegin(), True), ((1, 3, 5), QuarterBegin(), False), ((1, 12, 1), QuarterEnd(), False), ((1, 1, 1), YearBegin(), True), ((1, 1, 1, 1), YearBegin(), True), ((1, 1, 5), YearBegin(), False), ((1, 12, 1), YearEnd(), False), ((1, 1, 1), Day(), True), ((1, 1, 1, 1), Day(), True), ((1, 1, 1), Hour(), True), ((1, 1, 1), Minute(), True), ((1, 1, 1), Second(), True), ((1, 1, 1), Millisecond(), True), ((1, 1, 1), Microsecond(), True)], ids=_id_func)
def test_onOffset(calendar, date_args, offset, expected):
    date_type = get_date_type(calendar)
    date = date_type(*date_args)
    result = offset.onOffset(date)
    assert result == expected