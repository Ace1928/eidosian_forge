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
@pytest.mark.parametrize('calendar', _CFTIME_CALENDARS)
def test_pickle_cftimeindex(calendar):
    idx = xr.cftime_range('2000-01-01', periods=3, freq='D', calendar=calendar)
    idx_pkl = pickle.loads(pickle.dumps(idx))
    assert (idx == idx_pkl).all()