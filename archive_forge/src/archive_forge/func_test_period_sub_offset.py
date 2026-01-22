from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
def test_period_sub_offset(self):
    msg = '|'.join(['Input has different freq', 'Input cannot be converted to Period'])
    for freq in ['Y', '2Y', '3Y']:
        per = Period('2011', freq=freq)
        assert per - offsets.YearEnd(2) == Period('2009', freq=freq)
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(365, 'D'), timedelta(365)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per - off
    for freq in ['M', '2M', '3M']:
        per = Period('2011-03', freq=freq)
        assert per - offsets.MonthEnd(2) == Period('2011-01', freq=freq)
        assert per - offsets.MonthEnd(12) == Period('2010-03', freq=freq)
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(365, 'D'), timedelta(365)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per - off
    for freq in ['D', '2D', '3D']:
        per = Period('2011-04-01', freq=freq)
        assert per - offsets.Day(5) == Period('2011-03-27', freq=freq)
        assert per - offsets.Hour(24) == Period('2011-03-31', freq=freq)
        assert per - np.timedelta64(2, 'D') == Period('2011-03-30', freq=freq)
        assert per - np.timedelta64(3600 * 24, 's') == Period('2011-03-31', freq=freq)
        assert per - timedelta(-2) == Period('2011-04-03', freq=freq)
        assert per - timedelta(hours=48) == Period('2011-03-30', freq=freq)
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(4, 'h'), timedelta(hours=23)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per - off
    for freq in ['h', '2h', '3h']:
        per = Period('2011-04-01 09:00', freq=freq)
        assert per - offsets.Day(2) == Period('2011-03-30 09:00', freq=freq)
        assert per - offsets.Hour(3) == Period('2011-04-01 06:00', freq=freq)
        assert per - np.timedelta64(3, 'h') == Period('2011-04-01 06:00', freq=freq)
        assert per - np.timedelta64(3600, 's') == Period('2011-04-01 08:00', freq=freq)
        assert per - timedelta(minutes=120) == Period('2011-04-01 07:00', freq=freq)
        assert per - timedelta(days=4, minutes=180) == Period('2011-03-28 06:00', freq=freq)
        for off in [offsets.YearBegin(2), offsets.MonthBegin(1), offsets.Minute(), np.timedelta64(3200, 's'), timedelta(hours=23, minutes=30)]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                per - off