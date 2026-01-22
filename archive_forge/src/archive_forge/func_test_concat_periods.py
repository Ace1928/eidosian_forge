from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
def test_concat_periods(self):
    periods = pd.period_range('2000-01-01', periods=10)
    indexes = [PandasIndex(periods[:5], 't'), PandasIndex(periods[5:], 't')]
    expected = PandasIndex(periods, 't')
    actual = PandasIndex.concat(indexes, dim='t')
    assert actual.equals(expected)
    assert isinstance(actual.index, pd.PeriodIndex)
    positions = [list(range(5)), list(range(5, 10))]
    actual = PandasIndex.concat(indexes, dim='t', positions=positions)
    assert actual.equals(expected)
    assert isinstance(actual.index, pd.PeriodIndex)