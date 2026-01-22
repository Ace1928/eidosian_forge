import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_asof_nanosecond_index_access(self):
    ts = Timestamp('20130101').as_unit('ns')._value
    dti = DatetimeIndex([ts + 50 + i for i in range(100)])
    ser = Series(np.random.default_rng(2).standard_normal(100), index=dti)
    first_value = ser.asof(ser.index[0])
    assert dti.resolution == 'nanosecond'
    assert first_value == ser['2013-01-01 00:00:00.000000050']
    expected_ts = np.datetime64('2013-01-01 00:00:00.000000050', 'ns')
    assert first_value == ser[Timestamp(expected_ts)]