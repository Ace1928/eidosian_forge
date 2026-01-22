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
def test_format_timestamp_out_of_bounds(self) -> None:
    from datetime import datetime
    date = datetime(1300, 12, 1)
    expected = '1300-12-01'
    result = formatting.format_timestamp(date)
    assert result == expected
    date = datetime(2300, 12, 1)
    expected = '2300-12-01'
    result = formatting.format_timestamp(date)
    assert result == expected