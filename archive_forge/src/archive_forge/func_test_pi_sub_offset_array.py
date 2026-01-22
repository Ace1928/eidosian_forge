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
@pytest.mark.parametrize('box', [np.array, pd.Index])
def test_pi_sub_offset_array(self, box):
    pi = PeriodIndex([Period('2015Q1'), Period('2016Q2')])
    other = box([pd.offsets.QuarterEnd(n=1, startingMonth=12), pd.offsets.QuarterEnd(n=-2, startingMonth=12)])
    expected = PeriodIndex([pi[n] - other[n] for n in range(len(pi))])
    expected = expected.astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res = pi - other
    tm.assert_index_equal(res, expected)
    anchored = box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)])
    msg = 'Input has different freq=-1M from Period\\(freq=Q-DEC\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            pi - anchored
    with pytest.raises(IncompatibleFrequency, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            anchored - pi