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
@pytest.mark.parametrize('freq', ['3663s', '33min', '2h'])
@pytest.mark.parametrize('method', ['floor', 'ceil', 'round'])
def test_rounding_methods_against_datetimeindex(freq, method):
    expected = pd.date_range('2000-01-02T01:03:51', periods=10, freq='1777s')
    expected = getattr(expected, method)(freq)
    result = xr.cftime_range('2000-01-02T01:03:51', periods=10, freq='1777s')
    result = getattr(result, method)(freq).to_datetimeindex()
    assert result.equals(expected)