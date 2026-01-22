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
@pytest.mark.parametrize('sel_kwargs', [{'method': 'pad', 'tolerance': timedelta(days=20)}, {'method': 'backfill', 'tolerance': timedelta(days=20)}, {'method': 'nearest', 'tolerance': timedelta(days=20)}])
def test_sel_date_scalar_tolerance_raises(da, date_type, sel_kwargs):
    with pytest.raises(KeyError):
        da.sel(time=date_type(1, 5, 1), **sel_kwargs)