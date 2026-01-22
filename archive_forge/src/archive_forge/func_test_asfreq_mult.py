import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_asfreq_mult(self):
    p = Period(freq='Y', year=2007)
    for freq in ['3Y', offsets.YearEnd(3)]:
        result = p.asfreq(freq)
        expected = Period('2007', freq='3Y')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    for freq in ['3Y', offsets.YearEnd(3)]:
        result = p.asfreq(freq, how='S')
        expected = Period('2007', freq='3Y')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    p = Period(freq='3Y', year=2007)
    for freq in ['Y', offsets.YearEnd()]:
        result = p.asfreq(freq)
        expected = Period('2009', freq='Y')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    for freq in ['Y', offsets.YearEnd()]:
        result = p.asfreq(freq, how='s')
        expected = Period('2007', freq='Y')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    p = Period(freq='Y', year=2007)
    for freq in ['2M', offsets.MonthEnd(2)]:
        result = p.asfreq(freq)
        expected = Period('2007-12', freq='2M')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    for freq in ['2M', offsets.MonthEnd(2)]:
        result = p.asfreq(freq, how='s')
        expected = Period('2007-01', freq='2M')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    p = Period(freq='3Y', year=2007)
    for freq in ['2M', offsets.MonthEnd(2)]:
        result = p.asfreq(freq)
        expected = Period('2009-12', freq='2M')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq
    for freq in ['2M', offsets.MonthEnd(2)]:
        result = p.asfreq(freq, how='s')
        expected = Period('2007-01', freq='2M')
        assert result == expected
        assert result.ordinal == expected.ordinal
        assert result.freq == expected.freq