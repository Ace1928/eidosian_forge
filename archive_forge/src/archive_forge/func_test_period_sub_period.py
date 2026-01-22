from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_sub_period(self):
    per1 = Period('2011-01-01', freq='D')
    per2 = Period('2011-01-15', freq='D')
    off = per1.freq
    assert per1 - per2 == -14 * off
    assert per2 - per1 == 14 * off
    msg = 'Input has different freq=M from Period\\(freq=D\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        per1 - Period('2011-02', freq='M')