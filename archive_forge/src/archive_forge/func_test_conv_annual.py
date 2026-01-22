import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_annual(self):
    ival_A = Period(freq='Y', year=2007)
    ival_AJAN = Period(freq='Y-JAN', year=2007)
    ival_AJUN = Period(freq='Y-JUN', year=2007)
    ival_ANOV = Period(freq='Y-NOV', year=2007)
    ival_A_to_Q_start = Period(freq='Q', year=2007, quarter=1)
    ival_A_to_Q_end = Period(freq='Q', year=2007, quarter=4)
    ival_A_to_M_start = Period(freq='M', year=2007, month=1)
    ival_A_to_M_end = Period(freq='M', year=2007, month=12)
    ival_A_to_W_start = Period(freq='W', year=2007, month=1, day=1)
    ival_A_to_W_end = Period(freq='W', year=2007, month=12, day=31)
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        ival_A_to_B_start = Period(freq='B', year=2007, month=1, day=1)
        ival_A_to_B_end = Period(freq='B', year=2007, month=12, day=31)
    ival_A_to_D_start = Period(freq='D', year=2007, month=1, day=1)
    ival_A_to_D_end = Period(freq='D', year=2007, month=12, day=31)
    ival_A_to_H_start = Period(freq='h', year=2007, month=1, day=1, hour=0)
    ival_A_to_H_end = Period(freq='h', year=2007, month=12, day=31, hour=23)
    ival_A_to_T_start = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0)
    ival_A_to_T_end = Period(freq='Min', year=2007, month=12, day=31, hour=23, minute=59)
    ival_A_to_S_start = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=0, second=0)
    ival_A_to_S_end = Period(freq='s', year=2007, month=12, day=31, hour=23, minute=59, second=59)
    ival_AJAN_to_D_end = Period(freq='D', year=2007, month=1, day=31)
    ival_AJAN_to_D_start = Period(freq='D', year=2006, month=2, day=1)
    ival_AJUN_to_D_end = Period(freq='D', year=2007, month=6, day=30)
    ival_AJUN_to_D_start = Period(freq='D', year=2006, month=7, day=1)
    ival_ANOV_to_D_end = Period(freq='D', year=2007, month=11, day=30)
    ival_ANOV_to_D_start = Period(freq='D', year=2006, month=12, day=1)
    assert ival_A.asfreq('Q', 's') == ival_A_to_Q_start
    assert ival_A.asfreq('Q', 'e') == ival_A_to_Q_end
    assert ival_A.asfreq('M', 's') == ival_A_to_M_start
    assert ival_A.asfreq('M', 'E') == ival_A_to_M_end
    assert ival_A.asfreq('W', 's') == ival_A_to_W_start
    assert ival_A.asfreq('W', 'E') == ival_A_to_W_end
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert ival_A.asfreq('B', 's') == ival_A_to_B_start
        assert ival_A.asfreq('B', 'E') == ival_A_to_B_end
    assert ival_A.asfreq('D', 's') == ival_A_to_D_start
    assert ival_A.asfreq('D', 'E') == ival_A_to_D_end
    msg = "'H' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ival_A.asfreq('H', 's') == ival_A_to_H_start
        assert ival_A.asfreq('H', 'E') == ival_A_to_H_end
    assert ival_A.asfreq('min', 's') == ival_A_to_T_start
    assert ival_A.asfreq('min', 'E') == ival_A_to_T_end
    msg = "'T' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ival_A.asfreq('T', 's') == ival_A_to_T_start
        assert ival_A.asfreq('T', 'E') == ival_A_to_T_end
    msg = "'S' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ival_A.asfreq('S', 'S') == ival_A_to_S_start
        assert ival_A.asfreq('S', 'E') == ival_A_to_S_end
    assert ival_AJAN.asfreq('D', 's') == ival_AJAN_to_D_start
    assert ival_AJAN.asfreq('D', 'E') == ival_AJAN_to_D_end
    assert ival_AJUN.asfreq('D', 's') == ival_AJUN_to_D_start
    assert ival_AJUN.asfreq('D', 'E') == ival_AJUN_to_D_end
    assert ival_ANOV.asfreq('D', 's') == ival_ANOV_to_D_start
    assert ival_ANOV.asfreq('D', 'E') == ival_ANOV_to_D_end
    assert ival_A.asfreq('Y') == ival_A