from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_ops_offset(self):
    per = Period('2011-04-01', freq='D')
    result = per + offsets.Day()
    exp = Period('2011-04-02', freq='D')
    assert result == exp
    result = per - offsets.Day(2)
    exp = Period('2011-03-30', freq='D')
    assert result == exp
    msg = 'Input cannot be converted to Period\\(freq=D\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        per + offsets.Hour(2)
    with pytest.raises(IncompatibleFrequency, match=msg):
        per - offsets.Hour(2)