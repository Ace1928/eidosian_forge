from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props
from pandas.tseries.offsets import WeekOfMonth
def test_get_finder(self):
    import pandas.plotting._matplotlib.converter as conv
    assert conv.get_finder(to_offset('B')) == conv._daily_finder
    assert conv.get_finder(to_offset('D')) == conv._daily_finder
    assert conv.get_finder(to_offset('ME')) == conv._monthly_finder
    assert conv.get_finder(to_offset('QE')) == conv._quarterly_finder
    assert conv.get_finder(to_offset('YE')) == conv._annual_finder
    assert conv.get_finder(to_offset('W')) == conv._daily_finder