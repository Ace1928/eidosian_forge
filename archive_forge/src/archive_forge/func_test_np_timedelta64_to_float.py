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
@pytest.mark.parametrize('td, expected', ([np.timedelta64(1, 'D'), 86400 * 1000000000.0], [np.timedelta64(1, 'ns'), 1.0]))
def test_np_timedelta64_to_float(td, expected):
    out = np_timedelta64_to_float(td, datetime_unit='ns')
    np.testing.assert_allclose(out, expected)
    assert isinstance(out, float)
    out = np_timedelta64_to_float(np.atleast_1d(td), datetime_unit='ns')
    np.testing.assert_allclose(out, expected)