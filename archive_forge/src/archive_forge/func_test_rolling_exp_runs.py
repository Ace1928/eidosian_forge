from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('dim', ['time', 'x'])
@pytest.mark.parametrize('window_type, window', [['span', 5], ['alpha', 0.5], ['com', 0.5], ['halflife', 5]])
@pytest.mark.parametrize('backend', ['numpy'], indirect=True)
@pytest.mark.parametrize('func', ['mean', 'sum', 'var', 'std'])
def test_rolling_exp_runs(self, da, dim, window_type, window, func) -> None:
    da = da.where(da > 0.2)
    rolling_exp = da.rolling_exp(window_type=window_type, **{dim: window})
    result = getattr(rolling_exp, func)()
    assert isinstance(result, DataArray)