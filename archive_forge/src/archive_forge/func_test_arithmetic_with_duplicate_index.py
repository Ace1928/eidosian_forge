from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_arithmetic_with_duplicate_index(self):
    index = [2, 2, 3, 3, 4]
    ser = Series(np.arange(1, 6, dtype='int64'), index=index)
    other = Series(np.arange(5, dtype='int64'), index=index)
    result = ser - other
    expected = Series(1, index=[2, 2, 3, 3, 4])
    tm.assert_series_equal(result, expected)
    ser = Series(date_range('20130101 09:00:00', periods=5), index=index)
    other = Series(date_range('20130101', periods=5), index=index)
    result = ser - other
    expected = Series(Timedelta('9 hours'), index=[2, 2, 3, 3, 4])
    tm.assert_series_equal(result, expected)