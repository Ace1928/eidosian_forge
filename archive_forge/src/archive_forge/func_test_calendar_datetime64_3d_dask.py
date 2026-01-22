from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@requires_dask
def test_calendar_datetime64_3d_dask() -> None:
    import dask.array as da
    data = xr.DataArray(da.zeros((4, 5, 6), dtype='datetime64[ns]'), dims=('x', 'y', 'z'))
    with raise_if_dask_computes():
        assert data.dt.calendar == 'proleptic_gregorian'