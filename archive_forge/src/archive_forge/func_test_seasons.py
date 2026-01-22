from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
def test_seasons(self) -> None:
    dates = xr.date_range(start='2000/01/01', freq='ME', periods=12, use_cftime=False)
    dates = dates.append(pd.Index([np.datetime64('NaT')]))
    dates = xr.DataArray(dates)
    seasons = xr.DataArray(['DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF', 'nan'])
    assert_array_equal(seasons.values, dates.dt.season.values)