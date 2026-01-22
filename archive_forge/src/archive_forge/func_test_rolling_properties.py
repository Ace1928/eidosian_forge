from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
def test_rolling_properties(self, ds) -> None:
    with pytest.raises(ValueError, match='window must be > 0'):
        ds.rolling(time=-2)
    with pytest.raises(ValueError, match='min_periods must be greater than zero'):
        ds.rolling(time=2, min_periods=0)
    with pytest.raises(KeyError, match='time2'):
        ds.rolling(time2=2)
    with pytest.raises(KeyError, match="\\('foo',\\) not found in Dataset dimensions"):
        ds.rolling(foo=2)