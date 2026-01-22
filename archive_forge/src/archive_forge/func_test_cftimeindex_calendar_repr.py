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
@pytest.mark.parametrize(('calendar', 'expected'), [('noleap', 'noleap'), ('365_day', 'noleap'), ('360_day', '360_day'), ('julian', 'julian'), ('gregorian', standard_or_gregorian), ('standard', standard_or_gregorian), ('proleptic_gregorian', 'proleptic_gregorian')])
def test_cftimeindex_calendar_repr(calendar, expected):
    """Test that cftimeindex has calendar property in repr."""
    index = xr.cftime_range(start='2000', periods=3, calendar=calendar)
    repr_str = index.__repr__()
    assert f" calendar='{expected}'" in repr_str
    assert '2000-01-01 00:00:00, 2000-01-02 00:00:00' in repr_str