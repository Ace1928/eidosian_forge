from __future__ import annotations
import datetime
from typing import TypedDict
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core.pdcompat import _convert_base_to_offset
from xarray.core.resample_cftime import CFTimeGrouper
def test_timedelta_offset() -> None:
    timedelta = datetime.timedelta(seconds=5)
    string = '5s'
    cftime_index = xr.cftime_range('2000', periods=5)
    da_cftime = da(cftime_index)
    timedelta_result = da_cftime.resample(time='2D', offset=timedelta).mean()
    string_result = da_cftime.resample(time='2D', offset=string).mean()
    xr.testing.assert_identical(timedelta_result, string_result)