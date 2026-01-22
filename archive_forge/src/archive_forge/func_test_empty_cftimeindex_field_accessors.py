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
@pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'dayofyear', 'dayofweek', 'days_in_month'])
def test_empty_cftimeindex_field_accessors(field):
    index = CFTimeIndex([])
    result = getattr(index, field)
    expected = np.array([], dtype=np.int64)
    assert_array_equal(result, expected)
    assert result.dtype == expected.dtype