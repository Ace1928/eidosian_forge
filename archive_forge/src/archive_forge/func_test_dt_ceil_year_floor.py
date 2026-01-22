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
@pytest.mark.parametrize('freq', ['D', 'h', 'min', 's', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('method', ['ceil', 'floor', 'round'])
def test_dt_ceil_year_floor(freq, method):
    ser = pd.Series([datetime(year=2023, month=1, day=1), None])
    pa_dtype = ArrowDtype(pa.timestamp('ns'))
    expected = getattr(ser.dt, method)(f'1{freq}').astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f'1{freq}')
    tm.assert_series_equal(result, expected)