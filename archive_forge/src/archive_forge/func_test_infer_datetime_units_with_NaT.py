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
@pytest.mark.parametrize(['dates', 'expected'], [(pd.to_datetime(['1900-01-01', '1900-01-02', 'NaT']), 'days since 1900-01-01 00:00:00'), (pd.to_datetime(['NaT', '1900-01-01']), 'days since 1900-01-01 00:00:00'), (pd.to_datetime(['NaT']), 'days since 1970-01-01 00:00:00')])
def test_infer_datetime_units_with_NaT(dates, expected) -> None:
    assert expected == coding.times.infer_datetime_units(dates)