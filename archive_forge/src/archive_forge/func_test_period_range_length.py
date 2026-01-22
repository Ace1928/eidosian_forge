import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_period_range_length(self):
    pi = period_range(freq='Y', start='1/1/2001', end='12/1/2009')
    assert len(pi) == 9
    pi = period_range(freq='Q', start='1/1/2001', end='12/1/2009')
    assert len(pi) == 4 * 9
    pi = period_range(freq='M', start='1/1/2001', end='12/1/2009')
    assert len(pi) == 12 * 9
    pi = period_range(freq='D', start='1/1/2001', end='12/31/2009')
    assert len(pi) == 365 * 9 + 2
    msg = 'Period with BDay freq is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        pi = period_range(freq='B', start='1/1/2001', end='12/31/2009')
    assert len(pi) == 261 * 9
    pi = period_range(freq='h', start='1/1/2001', end='12/31/2001 23:00')
    assert len(pi) == 365 * 24
    pi = period_range(freq='Min', start='1/1/2001', end='1/1/2001 23:59')
    assert len(pi) == 24 * 60
    pi = period_range(freq='s', start='1/1/2001', end='1/1/2001 23:59:59')
    assert len(pi) == 24 * 60 * 60
    with tm.assert_produces_warning(FutureWarning, match=msg):
        start = Period('02-Apr-2005', 'B')
        i1 = period_range(start=start, periods=20)
    assert len(i1) == 20
    assert i1.freq == start.freq
    assert i1[0] == start
    end_intv = Period('2006-12-31', 'W')
    i1 = period_range(end=end_intv, periods=10)
    assert len(i1) == 10
    assert i1.freq == end_intv.freq
    assert i1[-1] == end_intv
    msg = "'w' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        end_intv = Period('2006-12-31', '1w')
    i2 = period_range(end=end_intv, periods=10)
    assert len(i1) == len(i2)
    assert (i1 == i2).all()
    assert i1.freq == i2.freq