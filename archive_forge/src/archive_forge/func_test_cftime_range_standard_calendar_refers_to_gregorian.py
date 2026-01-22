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
def test_cftime_range_standard_calendar_refers_to_gregorian() -> None:
    from cftime import DatetimeGregorian
    result, = cftime_range('2000', periods=1)
    assert isinstance(result, DatetimeGregorian)