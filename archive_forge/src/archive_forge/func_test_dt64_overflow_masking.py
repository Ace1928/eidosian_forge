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
def test_dt64_overflow_masking(self, box_with_array):
    left = Series([Timestamp('1969-12-31')], dtype='M8[ns]')
    right = Series([NaT])
    left = tm.box_expected(left, box_with_array)
    right = tm.box_expected(right, box_with_array)
    expected = TimedeltaIndex([NaT], dtype='m8[ns]')
    expected = tm.box_expected(expected, box_with_array)
    result = left - right
    tm.assert_equal(result, expected)