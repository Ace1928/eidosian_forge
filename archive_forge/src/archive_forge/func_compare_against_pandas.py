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
def compare_against_pandas(da_datetimeindex, da_cftimeindex, freq, closed=None, label=None, base=None, offset=None, origin=None, loffset=None) -> None:
    if isinstance(origin, tuple):
        origin_pandas = pd.Timestamp(datetime.datetime(*origin))
        origin_cftime = cftime.DatetimeGregorian(*origin)
    else:
        origin_pandas = origin
        origin_cftime = origin
    try:
        result_datetimeindex = da_datetimeindex.resample(time=freq, closed=closed, label=label, base=base, loffset=loffset, offset=offset, origin=origin_pandas).mean()
    except ValueError:
        with pytest.raises(ValueError):
            da_cftimeindex.resample(time=freq, closed=closed, label=label, base=base, loffset=loffset, origin=origin_cftime, offset=offset).mean()
    else:
        result_cftimeindex = da_cftimeindex.resample(time=freq, closed=closed, label=label, base=base, loffset=loffset, origin=origin_cftime, offset=offset).mean()
    result_cftimeindex['time'] = result_cftimeindex.xindexes['time'].to_pandas_index().to_datetimeindex()
    xr.testing.assert_identical(result_cftimeindex, result_datetimeindex)