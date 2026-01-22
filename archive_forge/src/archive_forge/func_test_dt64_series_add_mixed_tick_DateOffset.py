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
def test_dt64_series_add_mixed_tick_DateOffset(self):
    s = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
    result = s + pd.offsets.Milli(5)
    result2 = pd.offsets.Milli(5) + s
    expected = Series([Timestamp('20130101 9:01:00.005'), Timestamp('20130101 9:02:00.005')])
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)
    result = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
    expected = Series([Timestamp('20130101 9:06:00.005'), Timestamp('20130101 9:07:00.005')])
    tm.assert_series_equal(result, expected)