import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_shift_gh8083(self):
    drange = period_range('20130101', periods=5, freq='D')
    result = drange.shift(1)
    expected = PeriodIndex(['2013-01-02', '2013-01-03', '2013-01-04', '2013-01-05', '2013-01-06'], freq='D')
    tm.assert_index_equal(result, expected)