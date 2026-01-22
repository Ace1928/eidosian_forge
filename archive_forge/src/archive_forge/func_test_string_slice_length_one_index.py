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
def test_string_slice_length_one_index(length_one_index):
    da = xr.DataArray([1], coords=[length_one_index], dims=['time'])
    result = da.sel(time=slice('0001', '0001'))
    assert_identical(result, da)