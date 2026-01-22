from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize('td', [dt.timedelta(days=1), np.timedelta64(1, 'D'), pd.Timedelta(1, 'D'), '1 day'])
def test_timedelta_to_numeric(td):
    out = timedelta_to_numeric(td, 'ns')
    np.testing.assert_allclose(out, 86400 * 1000000000.0)
    assert isinstance(out, float)