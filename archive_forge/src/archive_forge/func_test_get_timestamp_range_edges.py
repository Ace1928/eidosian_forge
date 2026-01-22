from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
@pytest.mark.parametrize('first,last,freq,exp_first,exp_last', [('19910905', '19920406', 'D', '19910905', '19920407'), ('19910905 00:00', '19920406 06:00', 'D', '19910905', '19920407'), ('19910905 06:00', '19920406 06:00', 'h', '19910905 06:00', '19920406 07:00'), ('19910906', '19920406', 'ME', '19910831', '19920430'), ('19910831', '19920430', 'ME', '19910831', '19920531'), ('1991-08', '1992-04', 'ME', '19910831', '19920531')])
def test_get_timestamp_range_edges(first, last, freq, exp_first, exp_last, unit):
    first = Period(first)
    first = first.to_timestamp(first.freq).as_unit(unit)
    last = Period(last)
    last = last.to_timestamp(last.freq).as_unit(unit)
    exp_first = Timestamp(exp_first)
    exp_last = Timestamp(exp_last)
    freq = pd.tseries.frequencies.to_offset(freq)
    result = _get_timestamp_range_edges(first, last, freq, unit='ns')
    expected = (exp_first, exp_last)
    assert result == expected