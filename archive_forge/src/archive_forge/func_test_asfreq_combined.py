import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_asfreq_combined(self):
    p = Period('2007', freq='h')
    expected = Period('2007', freq='25h')
    for freq, how in zip(['1D1h', '1h1D'], ['E', 'S']):
        result = p.asfreq(freq, how=how)
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    p1 = Period(freq='1D1h', year=2007)
    p2 = Period(freq='1h1D', year=2007)
    result1 = p1.asfreq('h')
    result2 = p2.asfreq('h')
    expected = Period('2007-01-02', freq='h')
    assert result1 == expected
    assert result1.ordinal == expected.ordinal
    assert result1.freq == expected.freq
    assert result2 == expected
    assert result2.ordinal == expected.ordinal
    assert result2.freq == expected.freq
    result1 = p1.asfreq('h', how='S')
    result2 = p2.asfreq('h', how='S')
    expected = Period('2007-01-01', freq='h')
    assert result1 == expected
    assert result1.ordinal == expected.ordinal
    assert result1.freq == expected.freq
    assert result2 == expected
    assert result2.ordinal == expected.ordinal
    assert result2.freq == expected.freq