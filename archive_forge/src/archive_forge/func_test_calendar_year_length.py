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
@pytest.mark.parametrize(('calendar', 'start', 'end', 'expected_number_of_days'), [('noleap', '2000', '2001', 365), ('all_leap', '2000', '2001', 366), ('360_day', '2000', '2001', 360), ('standard', '2000', '2001', 366), ('gregorian', '2000', '2001', 366), ('julian', '2000', '2001', 366), ('noleap', '2001', '2002', 365), ('all_leap', '2001', '2002', 366), ('360_day', '2001', '2002', 360), ('standard', '2001', '2002', 365), ('gregorian', '2001', '2002', 365), ('julian', '2001', '2002', 365)])
def test_calendar_year_length(calendar: str, start: str, end: str, expected_number_of_days: int) -> None:
    result = cftime_range(start, end, freq='D', inclusive='left', calendar=calendar)
    assert len(result) == expected_number_of_days