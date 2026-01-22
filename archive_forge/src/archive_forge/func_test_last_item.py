from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_last_item(self) -> None:
    array = np.arange(100)
    reshape = ((10, 10), (1, 100), (2, 2, 5, 5))
    expected = np.array([99])
    for r in reshape:
        result = formatting.last_item(array.reshape(r))
        assert result == expected