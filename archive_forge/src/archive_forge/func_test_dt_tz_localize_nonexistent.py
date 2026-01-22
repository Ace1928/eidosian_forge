from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
@pytest.mark.parametrize('nonexistent, exp_date', [['shift_forward', datetime(year=2023, month=3, day=12, hour=3)], ['shift_backward', pd.Timestamp('2023-03-12 01:59:59.999999999')]])
def test_dt_tz_localize_nonexistent(nonexistent, exp_date, request):
    _require_timezone_database(request)
    ser = pd.Series([datetime(year=2023, month=3, day=12, hour=2, minute=30), None], dtype=ArrowDtype(pa.timestamp('ns')))
    result = ser.dt.tz_localize('US/Pacific', nonexistent=nonexistent)
    exp_data = pa.array([exp_date, None], type=pa.timestamp('ns'))
    exp_data = pa.compute.assume_timezone(exp_data, 'US/Pacific')
    expected = pd.Series(ArrowExtensionArray(exp_data))
    tm.assert_series_equal(result, expected)