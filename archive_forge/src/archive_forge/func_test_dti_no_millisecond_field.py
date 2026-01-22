import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_dti_no_millisecond_field(self):
    msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"
    with pytest.raises(AttributeError, match=msg):
        DatetimeIndex.millisecond
    msg = "'DatetimeIndex' object has no attribute 'millisecond'"
    with pytest.raises(AttributeError, match=msg):
        DatetimeIndex([]).millisecond