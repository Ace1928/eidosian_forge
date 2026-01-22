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
@pytest.mark.parametrize(('base', 'freq'), [(1, '10s'), (17, '3h'), (15, '5us')])
def test__convert_base_to_offset(base, freq):
    datetimeindex = pd.date_range('2000', periods=2)
    cftimeindex = xr.cftime_range('2000', periods=2)
    pandas_result = _convert_base_to_offset(base, freq, datetimeindex)
    cftime_result = _convert_base_to_offset(base, freq, cftimeindex)
    assert pandas_result.to_pytimedelta() == cftime_result