from datetime import datetime
from dateutil.tz.tz import tzlocal
import pytest
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.tseries.offsets import (
def test_apply_out_of_range(request, tz_naive_fixture, _offset):
    tz = tz_naive_fixture
    try:
        if _offset in (BusinessHour, CustomBusinessHour):
            offset = _get_offset(_offset, value=100000)
        else:
            offset = _get_offset(_offset, value=10000)
        result = Timestamp('20080101') + offset
        assert isinstance(result, datetime)
        assert result.tzinfo is None
        t = Timestamp('20080101', tz=tz)
        result = t + offset
        assert isinstance(result, datetime)
        if tz is not None:
            assert t.tzinfo is not None
        if isinstance(tz, tzlocal) and (not IS64) and (_offset is not DateOffset):
            request.applymarker(pytest.mark.xfail(reason='OverflowError inside tzlocal past 2038'))
        elif isinstance(tz, tzlocal) and is_platform_windows() and (_offset in (QuarterEnd, BQuarterBegin, BQuarterEnd)):
            request.applymarker(pytest.mark.xfail(reason='After GH#49737 t.tzinfo is None on CI'))
        assert str(t.tzinfo) == str(result.tzinfo)
    except OutOfBoundsDatetime:
        pass
    except (ValueError, KeyError):
        pass