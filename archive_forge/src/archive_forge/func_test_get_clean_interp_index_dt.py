from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_cftime
@pytest.mark.parametrize(('calendar', 'freq'), zip(['gregorian', 'proleptic_gregorian'], ['1D', '1ME', '1Y']))
def test_get_clean_interp_index_dt(cf_da, calendar, freq):
    """In the gregorian case, the index should be proportional to normal datetimes."""
    g = cf_da(calendar, freq=freq)
    g['stime'] = xr.Variable(data=g.time.to_index().to_datetimeindex(), dims=('time',))
    gi = get_clean_interp_index(g, 'time')
    si = get_clean_interp_index(g, 'time', use_coordinate='stime')
    np.testing.assert_array_equal(gi, si)