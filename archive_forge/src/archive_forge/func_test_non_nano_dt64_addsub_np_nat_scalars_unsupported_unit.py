from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit():
    ser = Series([12332, 23243, 33243], dtype='datetime64[s]')
    result = ser - np.datetime64('nat', 'D')
    expected = Series([NaT] * 3, dtype='timedelta64[s]')
    tm.assert_series_equal(result, expected)
    result = ser + np.timedelta64('nat', 'D')
    expected = Series([NaT] * 3, dtype='datetime64[s]')
    tm.assert_series_equal(result, expected)