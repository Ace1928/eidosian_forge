from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_sub_period_annual(self):
    left, right = (Period('2011', freq='Y'), Period('2007', freq='Y'))
    result = left - right
    assert result == 4 * right.freq
    msg = 'Input has different freq=M from Period\\(freq=Y-DEC\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        left - Period('2007-01', freq='M')