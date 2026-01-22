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
def test_construct_from_string_invalid_raises(self):
    with pytest.raises(ValueError, match='gives an invalid tzoffset'):
        Timestamp('200622-12-31')