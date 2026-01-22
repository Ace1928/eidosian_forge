from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize(['deltas', 'expected'], [(pd.to_timedelta(['1 day', '2 days']), 'days'), (pd.to_timedelta(['1h', '1 day 1 hour']), 'hours'), (pd.to_timedelta(['1m', '2m', np.nan]), 'minutes'), (pd.to_timedelta(['1m3s', '1m4s']), 'seconds')])
def test_infer_timedelta_units(deltas, expected) -> None:
    assert expected == coding.times.infer_timedelta_units(deltas)