from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_construction_quarter(self):
    i1 = Period(year=2005, quarter=1, freq='Q')
    i2 = Period('1/1/2005', freq='Q')
    assert i1 == i2
    i1 = Period(year=2005, quarter=3, freq='Q')
    i2 = Period('9/1/2005', freq='Q')
    assert i1 == i2
    i1 = Period('2005Q1')
    i2 = Period(year=2005, quarter=1, freq='Q')
    i3 = Period('2005q1')
    assert i1 == i2
    assert i1 == i3
    i1 = Period('05Q1')
    assert i1 == i2
    lower = Period('05q1')
    assert i1 == lower
    i1 = Period('1Q2005')
    assert i1 == i2
    lower = Period('1q2005')
    assert i1 == lower
    i1 = Period('1Q05')
    assert i1 == i2
    lower = Period('1q05')
    assert i1 == lower
    i1 = Period('4Q1984')
    assert i1.year == 1984
    lower = Period('4q1984')
    assert i1 == lower