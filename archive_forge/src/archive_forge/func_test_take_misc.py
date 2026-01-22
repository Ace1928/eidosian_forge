from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_take_misc(self):
    index = period_range(start='1/1/10', end='12/31/12', freq='D', name='idx')
    expected = PeriodIndex([datetime(2010, 1, 6), datetime(2010, 1, 7), datetime(2010, 1, 9), datetime(2010, 1, 13)], freq='D', name='idx')
    taken1 = index.take([5, 6, 8, 12])
    taken2 = index[[5, 6, 8, 12]]
    for taken in [taken1, taken2]:
        tm.assert_index_equal(taken, expected)
        assert isinstance(taken, PeriodIndex)
        assert taken.freq == index.freq
        assert taken.name == expected.name