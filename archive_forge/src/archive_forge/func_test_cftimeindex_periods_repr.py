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
@pytest.mark.parametrize('periods', [2, 40])
def test_cftimeindex_periods_repr(periods):
    """Test that cftimeindex has periods property in repr."""
    index = xr.cftime_range(start='2000', periods=periods)
    repr_str = index.__repr__()
    assert f' length={periods}' in repr_str