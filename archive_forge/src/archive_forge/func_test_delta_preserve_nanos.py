from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_delta_preserve_nanos(self):
    val = Timestamp(1337299200000000123)
    result = val + timedelta(1)
    assert result.nanosecond == val.nanosecond