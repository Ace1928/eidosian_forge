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
def test_constructor_float_not_round_with_YM_unit_raises(self):
    msg = 'Conversion of non-round float with unit=[MY] is ambiguous'
    with pytest.raises(ValueError, match=msg):
        Timestamp(150.5, unit='Y')
    with pytest.raises(ValueError, match=msg):
        Timestamp(150.5, unit='M')