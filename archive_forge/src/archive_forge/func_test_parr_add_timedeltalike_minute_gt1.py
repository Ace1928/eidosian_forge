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
def test_parr_add_timedeltalike_minute_gt1(self, three_days, box_with_array):
    other = three_days
    rng = period_range('2014-05-01', periods=3, freq='2D')
    rng = tm.box_expected(rng, box_with_array)
    expected = PeriodIndex(['2014-05-04', '2014-05-06', '2014-05-08'], freq='2D')
    expected = tm.box_expected(expected, box_with_array)
    result = rng + other
    tm.assert_equal(result, expected)
    result = other + rng
    tm.assert_equal(result, expected)
    expected = PeriodIndex(['2014-04-28', '2014-04-30', '2014-05-02'], freq='2D')
    expected = tm.box_expected(expected, box_with_array)
    result = rng - other
    tm.assert_equal(result, expected)
    msg = '|'.join(["bad operand type for unary -: 'PeriodArray'", 'cannot subtract PeriodArray from timedelta64\\[[hD]\\]'])
    with pytest.raises(TypeError, match=msg):
        other - rng