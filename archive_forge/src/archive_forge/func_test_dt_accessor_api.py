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
def test_dt_accessor_api(self):
    from pandas.core.indexes.accessors import CombinedDatetimelikeProperties, DatetimeProperties
    assert Series.dt is CombinedDatetimelikeProperties
    ser = Series(date_range('2000-01-01', periods=3))
    assert isinstance(ser.dt, DatetimeProperties)