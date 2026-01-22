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
def test_days_in_month_december(calendar):
    if calendar == '360_day':
        expected = 30
    else:
        expected = 31
    date_type = get_date_type(calendar)
    reference = date_type(1, 12, 5)
    assert _days_in_month(reference) == expected