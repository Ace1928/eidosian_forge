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
def test_asi8(date_type):
    index = xr.CFTimeIndex([date_type(1970, 1, 1), date_type(1970, 1, 2)])
    result = index.asi8
    expected = 1000000 * 86400 * np.array([0, 1])
    np.testing.assert_array_equal(result, expected)