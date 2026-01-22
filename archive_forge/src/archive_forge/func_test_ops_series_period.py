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
def test_ops_series_period(self):
    ser = Series([Period('2015-01-01', freq='D'), Period('2015-01-02', freq='D')], name='xxx')
    assert ser.dtype == 'Period[D]'
    per = Period('2015-01-10', freq='D')
    off = per.freq
    expected = Series([9 * off, 8 * off], name='xxx', dtype=object)
    tm.assert_series_equal(per - ser, expected)
    tm.assert_series_equal(ser - per, -1 * expected)
    s2 = Series([Period('2015-01-05', freq='D'), Period('2015-01-04', freq='D')], name='xxx')
    assert s2.dtype == 'Period[D]'
    expected = Series([4 * off, 2 * off], name='xxx', dtype=object)
    tm.assert_series_equal(s2 - ser, expected)
    tm.assert_series_equal(ser - s2, -1 * expected)