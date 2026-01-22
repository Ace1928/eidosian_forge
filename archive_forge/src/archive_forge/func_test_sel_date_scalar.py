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
def test_sel_date_scalar(da, date_type, index):
    expected = xr.DataArray(1).assign_coords(time=index[0])
    result = da.sel(time=date_type(1, 1, 1))
    assert_identical(result, expected)