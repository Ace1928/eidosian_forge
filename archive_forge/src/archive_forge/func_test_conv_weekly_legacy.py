import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_weekly_legacy(self):
    msg = INVALID_FREQ_ERR_MSG
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK', year=2007, month=1, day=1)
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK-SAT', year=2007, month=1, day=6)
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK-FRI', year=2007, month=1, day=5)
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK-THU', year=2007, month=1, day=4)
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK-WED', year=2007, month=1, day=3)
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK-TUE', year=2007, month=1, day=2)
    with pytest.raises(ValueError, match=msg):
        Period(freq='WK-MON', year=2007, month=1, day=1)