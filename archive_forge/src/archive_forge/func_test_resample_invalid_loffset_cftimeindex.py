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
def test_resample_invalid_loffset_cftimeindex() -> None:
    times = xr.cftime_range('2000-01-01', freq='6h', periods=10)
    da = xr.DataArray(np.arange(10), [('time', times)])
    with pytest.raises(ValueError):
        da.resample(time='24h', loffset=1)