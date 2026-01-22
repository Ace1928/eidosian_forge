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
@pytest.mark.parametrize('kwargs', [{}, {'year': 2020}, {'year': 2020, 'month': 1}])
def test_constructor_missing_keyword(self, kwargs):
    msg1 = "function missing required argument '(year|month|day)' \\(pos [123]\\)"
    msg2 = "Required argument '(year|month|day)' \\(pos [123]\\) not found"
    msg = '|'.join([msg1, msg2])
    with pytest.raises(TypeError, match=msg):
        Timestamp(**kwargs)