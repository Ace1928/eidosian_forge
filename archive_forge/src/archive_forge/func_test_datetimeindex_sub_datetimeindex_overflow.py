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
def test_datetimeindex_sub_datetimeindex_overflow(self):
    dtimax = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
    dtimin = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
    ts_neg = pd.to_datetime(['1950-01-01', '1950-01-01']).as_unit('ns')
    ts_pos = pd.to_datetime(['1980-01-01', '1980-01-01']).as_unit('ns')
    expected = Timestamp.max._value - ts_pos[1]._value
    result = dtimax - ts_pos
    assert result[1]._value == expected
    expected = Timestamp.min._value - ts_neg[1]._value
    result = dtimin - ts_neg
    assert result[1]._value == expected
    msg = 'Overflow in int64 addition'
    with pytest.raises(OverflowError, match=msg):
        dtimax - ts_neg
    with pytest.raises(OverflowError, match=msg):
        dtimin - ts_pos
    tmin = pd.to_datetime([Timestamp.min])
    t1 = tmin + Timedelta.max + Timedelta('1us')
    with pytest.raises(OverflowError, match=msg):
        t1 - tmin
    tmax = pd.to_datetime([Timestamp.max])
    t2 = tmax + Timedelta.min - Timedelta('1us')
    with pytest.raises(OverflowError, match=msg):
        tmax - t2