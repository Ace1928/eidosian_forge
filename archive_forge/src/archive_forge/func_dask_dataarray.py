from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
@pytest.fixture
def dask_dataarray(dataarray: xr.DataArray) -> xr.DataArray:
    pytest.importorskip('dask')
    return dataarray.chunk()