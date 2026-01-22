import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dt_accessor_no_new_attributes(self):
    ser = Series(date_range('20130101', periods=5, freq='D'))
    with pytest.raises(AttributeError, match='You cannot add any new attribute'):
        ser.dt.xlabel = 'a'