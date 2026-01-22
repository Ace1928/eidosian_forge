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
def test_dt64tz_series_sub_dtitz(self):
    dti = date_range('1999-09-30', periods=10, tz='US/Pacific')
    ser = Series(dti)
    expected = Series(TimedeltaIndex(['0days'] * 10))
    res = dti - ser
    tm.assert_series_equal(res, expected)
    res = ser - dti
    tm.assert_series_equal(res, expected)