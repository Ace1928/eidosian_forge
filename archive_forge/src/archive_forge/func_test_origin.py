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
@pytest.mark.filterwarnings('ignore:.*the `(base|loffset)` parameter to resample')
@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('origin', ['start_day', 'start', 'end', 'end_day', 'epoch', (1970, 1, 1, 3, 2)], ids=lambda x: f'{x}')
def test_origin(closed, origin) -> None:
    initial_freq, resample_freq = ('3h', '9h')
    start = '1969-12-31T12:07:01'
    index_kwargs: DateRangeKwargs = dict(start=start, periods=12, freq=initial_freq)
    datetime_index = pd.date_range(**index_kwargs)
    cftime_index = xr.cftime_range(**index_kwargs)
    da_datetimeindex = da(datetime_index)
    da_cftimeindex = da(cftime_index)
    compare_against_pandas(da_datetimeindex, da_cftimeindex, resample_freq, closed=closed, origin=origin)