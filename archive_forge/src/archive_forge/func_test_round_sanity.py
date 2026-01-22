from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.xfail(reason='Failing on builds', strict=False)
@given(val=st.integers(iNaT + 1, lib.i8max))
@pytest.mark.parametrize('method', [Timestamp.round, Timestamp.floor, Timestamp.ceil])
def test_round_sanity(self, val, method):
    val = np.int64(val)
    ts = Timestamp(val)

    def checker(res, ts, nanos):
        if method is Timestamp.round:
            diff = np.abs((res - ts)._value)
            assert diff <= nanos / 2
        elif method is Timestamp.floor:
            assert res <= ts
        elif method is Timestamp.ceil:
            assert res >= ts
    assert method(ts, 'ns') == ts
    res = method(ts, 'us')
    nanos = 1000
    assert np.abs((res - ts)._value) < nanos
    assert res._value % nanos == 0
    checker(res, ts, nanos)
    res = method(ts, 'ms')
    nanos = 1000000
    assert np.abs((res - ts)._value) < nanos
    assert res._value % nanos == 0
    checker(res, ts, nanos)
    res = method(ts, 's')
    nanos = 1000000000
    assert np.abs((res - ts)._value) < nanos
    assert res._value % nanos == 0
    checker(res, ts, nanos)
    res = method(ts, 'min')
    nanos = 60 * 1000000000
    assert np.abs((res - ts)._value) < nanos
    assert res._value % nanos == 0
    checker(res, ts, nanos)
    res = method(ts, 'h')
    nanos = 60 * 60 * 1000000000
    assert np.abs((res - ts)._value) < nanos
    assert res._value % nanos == 0
    checker(res, ts, nanos)
    res = method(ts, 'D')
    nanos = 24 * 60 * 60 * 1000000000
    assert np.abs((res - ts)._value) < nanos
    assert res._value % nanos == 0
    checker(res, ts, nanos)