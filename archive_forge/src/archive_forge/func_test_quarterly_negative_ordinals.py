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
def test_quarterly_negative_ordinals(self):
    p = Period(ordinal=-1, freq='Q-DEC')
    assert p.year == 1969
    assert p.quarter == 4
    assert isinstance(p, Period)
    p = Period(ordinal=-2, freq='Q-DEC')
    assert p.year == 1969
    assert p.quarter == 3
    assert isinstance(p, Period)
    p = Period(ordinal=-2, freq='M')
    assert p.year == 1969
    assert p.month == 11
    assert isinstance(p, Period)