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
@pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
@pytest.mark.parametrize('tdi_freq', [None, 'h'])
def test_parr_sub_td64array(self, box_with_array, tdi_freq, pi_freq):
    box = box_with_array
    xbox = box if box not in [pd.array, tm.to_array] else pd.Index
    tdi = TimedeltaIndex(['1 hours', '2 hours'], freq=tdi_freq)
    dti = Timestamp('2018-03-07 17:16:40') + tdi
    pi = dti.to_period(pi_freq)
    td64obj = tm.box_expected(tdi, box)
    if pi_freq == 'h':
        result = pi - td64obj
        expected = (pi.to_timestamp('s') - tdi).to_period(pi_freq)
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        result = pi[0] - td64obj
        expected = (pi[0].to_timestamp('s') - tdi).to_period(pi_freq)
        expected = tm.box_expected(expected, box)
        tm.assert_equal(result, expected)
    elif pi_freq == 'D':
        msg = "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."
        with pytest.raises(IncompatibleFrequency, match=msg):
            pi - td64obj
        with pytest.raises(IncompatibleFrequency, match=msg):
            pi[0] - td64obj
    else:
        msg = 'Cannot add or subtract timedelta64'
        with pytest.raises(TypeError, match=msg):
            pi - td64obj
        with pytest.raises(TypeError, match=msg):
            pi[0] - td64obj