from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_period_cons_combined(self):
    p = [(Period('2011-01', freq='1D1h'), Period('2011-01', freq='1h1D'), Period('2011-01', freq='h')), (Period(ordinal=1, freq='1D1h'), Period(ordinal=1, freq='1h1D'), Period(ordinal=1, freq='h'))]
    for p1, p2, p3 in p:
        assert p1.ordinal == p3.ordinal
        assert p2.ordinal == p3.ordinal
        assert p1.freq == offsets.Hour(25)
        assert p1.freqstr == '25h'
        assert p2.freq == offsets.Hour(25)
        assert p2.freqstr == '25h'
        assert p3.freq == offsets.Hour()
        assert p3.freqstr == 'h'
        result = p1 + 1
        assert result.ordinal == (p3 + 25).ordinal
        assert result.freq == p1.freq
        assert result.freqstr == '25h'
        result = p2 + 1
        assert result.ordinal == (p3 + 25).ordinal
        assert result.freq == p2.freq
        assert result.freqstr == '25h'
        result = p1 - 1
        assert result.ordinal == (p3 - 25).ordinal
        assert result.freq == p1.freq
        assert result.freqstr == '25h'
        result = p2 - 1
        assert result.ordinal == (p3 - 25).ordinal
        assert result.freq == p2.freq
        assert result.freqstr == '25h'
    msg = 'Frequency must be positive, because it represents span: -25h'
    with pytest.raises(ValueError, match=msg):
        Period('2011-01', freq='-1D1h')
    with pytest.raises(ValueError, match=msg):
        Period('2011-01', freq='-1h1D')
    with pytest.raises(ValueError, match=msg):
        Period(ordinal=1, freq='-1D1h')
    with pytest.raises(ValueError, match=msg):
        Period(ordinal=1, freq='-1h1D')
    msg = 'Frequency must be positive, because it represents span: 0D'
    with pytest.raises(ValueError, match=msg):
        Period('2011-01', freq='0D0h')
    with pytest.raises(ValueError, match=msg):
        Period(ordinal=1, freq='0D0h')
    msg = 'Invalid frequency: 1W1D'
    with pytest.raises(ValueError, match=msg):
        Period('2011-01', freq='1W1D')
    msg = 'Invalid frequency: 1D1W'
    with pytest.raises(ValueError, match=msg):
        Period('2011-01', freq='1D1W')