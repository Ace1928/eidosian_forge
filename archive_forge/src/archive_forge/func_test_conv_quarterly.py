import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_quarterly(self):
    ival_Q = Period(freq='Q', year=2007, quarter=1)
    ival_Q_end_of_year = Period(freq='Q', year=2007, quarter=4)
    ival_QEJAN = Period(freq='Q-JAN', year=2007, quarter=1)
    ival_QEJUN = Period(freq='Q-JUN', year=2007, quarter=1)
    ival_Q_to_A = Period(freq='Y', year=2007)
    ival_Q_to_M_start = Period(freq='M', year=2007, month=1)
    ival_Q_to_M_end = Period(freq='M', year=2007, month=3)
    ival_Q_to_W_start = Period(freq='W', year=2007, month=1, day=1)
    ival_Q_to_W_end = Period(freq='W', year=2007, month=3, day=31)
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        ival_Q_to_B_start = Period(freq='B', year=2007, month=1, day=1)
        ival_Q_to_B_end = Period(freq='B', year=2007, month=3, day=30)
    ival_Q_to_D_start = Period(freq='D', year=2007, month=1, day=1)
    ival_Q_to_D_end = Period(freq='D', year=2007, month=3, day=31)
    ival_Q_to_H_start = Period(freq='h', year=2007, month=1, day=1, hour=0)
    ival_Q_to_H_end = Period(freq='h', year=2007, month=3, day=31, hour=23)
    ival_Q_to_T_start = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0)
    ival_Q_to_T_end = Period(freq='Min', year=2007, month=3, day=31, hour=23, minute=59)
    ival_Q_to_S_start = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=0, second=0)
    ival_Q_to_S_end = Period(freq='s', year=2007, month=3, day=31, hour=23, minute=59, second=59)
    ival_QEJAN_to_D_start = Period(freq='D', year=2006, month=2, day=1)
    ival_QEJAN_to_D_end = Period(freq='D', year=2006, month=4, day=30)
    ival_QEJUN_to_D_start = Period(freq='D', year=2006, month=7, day=1)
    ival_QEJUN_to_D_end = Period(freq='D', year=2006, month=9, day=30)
    assert ival_Q.asfreq('Y') == ival_Q_to_A
    assert ival_Q_end_of_year.asfreq('Y') == ival_Q_to_A
    assert ival_Q.asfreq('M', 's') == ival_Q_to_M_start
    assert ival_Q.asfreq('M', 'E') == ival_Q_to_M_end
    assert ival_Q.asfreq('W', 's') == ival_Q_to_W_start
    assert ival_Q.asfreq('W', 'E') == ival_Q_to_W_end
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert ival_Q.asfreq('B', 's') == ival_Q_to_B_start
        assert ival_Q.asfreq('B', 'E') == ival_Q_to_B_end
    assert ival_Q.asfreq('D', 's') == ival_Q_to_D_start
    assert ival_Q.asfreq('D', 'E') == ival_Q_to_D_end
    assert ival_Q.asfreq('h', 's') == ival_Q_to_H_start
    assert ival_Q.asfreq('h', 'E') == ival_Q_to_H_end
    assert ival_Q.asfreq('Min', 's') == ival_Q_to_T_start
    assert ival_Q.asfreq('Min', 'E') == ival_Q_to_T_end
    assert ival_Q.asfreq('s', 's') == ival_Q_to_S_start
    assert ival_Q.asfreq('s', 'E') == ival_Q_to_S_end
    assert ival_QEJAN.asfreq('D', 's') == ival_QEJAN_to_D_start
    assert ival_QEJAN.asfreq('D', 'E') == ival_QEJAN_to_D_end
    assert ival_QEJUN.asfreq('D', 's') == ival_QEJUN_to_D_start
    assert ival_QEJUN.asfreq('D', 'E') == ival_QEJUN_to_D_end
    assert ival_Q.asfreq('Q') == ival_Q