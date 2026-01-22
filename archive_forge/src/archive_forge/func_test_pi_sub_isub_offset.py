import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_pi_sub_isub_offset(self):
    rng = period_range('2014', '2024', freq='Y')
    result = rng - pd.offsets.YearEnd(5)
    expected = period_range('2009', '2019', freq='Y')
    tm.assert_index_equal(result, expected)
    rng -= pd.offsets.YearEnd(5)
    tm.assert_index_equal(rng, expected)
    rng = period_range('2014-01', '2016-12', freq='M')
    result = rng - pd.offsets.MonthEnd(5)
    expected = period_range('2013-08', '2016-07', freq='M')
    tm.assert_index_equal(result, expected)
    rng -= pd.offsets.MonthEnd(5)
    tm.assert_index_equal(rng, expected)