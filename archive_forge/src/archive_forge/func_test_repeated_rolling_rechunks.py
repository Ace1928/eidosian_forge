from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@requires_dask
def test_repeated_rolling_rechunks(self) -> None:
    dat = DataArray(np.random.rand(7653, 300), dims=('day', 'item'))
    dat_chunk = dat.chunk({'item': 20})
    dat_chunk.rolling(day=10).mean().rolling(day=250).std()