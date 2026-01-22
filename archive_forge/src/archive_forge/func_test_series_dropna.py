from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
def test_series_dropna(index):
    series = pd.Series([0.0, 1.0, np.nan, np.nan], index=index)
    expected = series.iloc[:2]
    result = series.dropna()
    assert result.equals(expected)