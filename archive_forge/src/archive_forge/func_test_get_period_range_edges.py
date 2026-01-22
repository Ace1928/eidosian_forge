from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
@pytest.mark.parametrize('first,last,freq,freq_to_offset,exp_first,exp_last', [('19910905', '19920406', 'D', 'D', '19910905', '19920406'), ('19910905 00:00', '19920406 06:00', 'D', 'D', '19910905', '19920406'), ('19910905 06:00', '19920406 06:00', 'h', 'h', '19910905 06:00', '19920406 06:00'), ('19910906', '19920406', 'M', 'ME', '1991-09', '1992-04'), ('19910831', '19920430', 'M', 'ME', '1991-08', '1992-04'), ('1991-08', '1992-04', 'M', 'ME', '1991-08', '1992-04')])
def test_get_period_range_edges(self, first, last, freq, freq_to_offset, exp_first, exp_last):
    first = Period(first)
    last = Period(last)
    exp_first = Period(exp_first, freq=freq)
    exp_last = Period(exp_last, freq=freq)
    freq = pd.tseries.frequencies.to_offset(freq_to_offset)
    result = _get_period_range_edges(first, last, freq)
    expected = (exp_first, exp_last)
    assert result == expected