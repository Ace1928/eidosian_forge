from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_tz_localize_ambiguous_bool(self, unit):
    ts = Timestamp('2015-11-01 01:00:03').as_unit(unit)
    expected0 = Timestamp('2015-11-01 01:00:03-0500', tz='US/Central')
    expected1 = Timestamp('2015-11-01 01:00:03-0600', tz='US/Central')
    msg = 'Cannot infer dst time from 2015-11-01 01:00:03'
    with pytest.raises(pytz.AmbiguousTimeError, match=msg):
        ts.tz_localize('US/Central')
    with pytest.raises(pytz.AmbiguousTimeError, match=msg):
        ts.tz_localize('dateutil/US/Central')
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo('US/Central')
        except KeyError:
            pass
        else:
            with pytest.raises(pytz.AmbiguousTimeError, match=msg):
                ts.tz_localize(tz)
    result = ts.tz_localize('US/Central', ambiguous=True)
    assert result == expected0
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
    result = ts.tz_localize('US/Central', ambiguous=False)
    assert result == expected1
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value