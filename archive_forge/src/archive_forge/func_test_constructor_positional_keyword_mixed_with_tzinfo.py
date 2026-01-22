import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('kwd', ['nanosecond', 'microsecond', 'second', 'minute'])
def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd, request):
    if kwd != 'nanosecond':
        mark = pytest.mark.xfail(reason='GH#45307')
        request.applymarker(mark)
    kwargs = {kwd: 4}
    ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)
    td_kwargs = {kwd + 's': 4}
    td = Timedelta(**td_kwargs)
    expected = Timestamp('2020-12-31', tz=timezone.utc) + td
    assert ts == expected