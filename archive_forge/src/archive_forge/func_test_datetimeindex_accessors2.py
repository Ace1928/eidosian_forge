import calendar
from datetime import datetime
import locale
import unicodedata
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tseries.frequencies import to_offset
def test_datetimeindex_accessors2(self):
    dti = date_range(freq='BQ-FEB', start=datetime(1998, 1, 1), periods=4)
    assert sum(dti.is_quarter_start) == 0
    assert sum(dti.is_quarter_end) == 4
    assert sum(dti.is_year_start) == 0
    assert sum(dti.is_year_end) == 1