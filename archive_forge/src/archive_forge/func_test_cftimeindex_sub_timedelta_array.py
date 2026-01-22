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
@pytest.mark.parametrize('other', [np.array(4 * [timedelta(days=1)]), np.array(timedelta(days=1))], ids=['1d-array', 'scalar-array'])
def test_cftimeindex_sub_timedelta_array(index, other):
    date_type = index.date_type
    expected_dates = [date_type(1, 1, 2), date_type(1, 2, 2), date_type(2, 1, 2), date_type(2, 2, 2)]
    expected = CFTimeIndex(expected_dates)
    result = index + timedelta(days=2)
    result = result - other
    assert result.equals(expected)
    assert isinstance(result, CFTimeIndex)