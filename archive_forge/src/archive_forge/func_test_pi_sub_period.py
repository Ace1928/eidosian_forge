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
def test_pi_sub_period(self):
    idx = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M', name='idx')
    result = idx - Period('2012-01', freq='M')
    off = idx.freq
    exp = pd.Index([-12 * off, -11 * off, -10 * off, -9 * off], name='idx')
    tm.assert_index_equal(result, exp)
    result = np.subtract(idx, Period('2012-01', freq='M'))
    tm.assert_index_equal(result, exp)
    result = Period('2012-01', freq='M') - idx
    exp = pd.Index([12 * off, 11 * off, 10 * off, 9 * off], name='idx')
    tm.assert_index_equal(result, exp)
    result = np.subtract(Period('2012-01', freq='M'), idx)
    tm.assert_index_equal(result, exp)
    exp = TimedeltaIndex([np.nan, np.nan, np.nan, np.nan], name='idx')
    result = idx - Period('NaT', freq='M')
    tm.assert_index_equal(result, exp)
    assert result.freq == exp.freq
    result = Period('NaT', freq='M') - idx
    tm.assert_index_equal(result, exp)
    assert result.freq == exp.freq