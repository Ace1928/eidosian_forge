import numpy as np
import pytest
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_tdi_round(self):
    td = timedelta_range(start='16801 days', periods=5, freq='30Min')
    elt = td[1]
    expected_rng = TimedeltaIndex([Timedelta('16801 days 00:00:00'), Timedelta('16801 days 00:00:00'), Timedelta('16801 days 01:00:00'), Timedelta('16801 days 02:00:00'), Timedelta('16801 days 02:00:00')])
    expected_elt = expected_rng[1]
    tm.assert_index_equal(td.round(freq='h'), expected_rng)
    assert elt.round(freq='h') == expected_elt
    msg = INVALID_FREQ_ERR_MSG
    with pytest.raises(ValueError, match=msg):
        td.round(freq='foo')
    with pytest.raises(ValueError, match=msg):
        elt.round(freq='foo')
    msg = '<MonthEnd> is a non-fixed frequency'
    with pytest.raises(ValueError, match=msg):
        td.round(freq='ME')
    with pytest.raises(ValueError, match=msg):
        elt.round(freq='ME')