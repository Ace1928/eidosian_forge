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
def test_construction_bday(self):
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        i1 = Period('3/10/12', freq='B')
        i2 = Period('3/10/12', freq='D')
        assert i1 == i2.asfreq('B')
        i2 = Period('3/11/12', freq='D')
        assert i1 == i2.asfreq('B')
        i2 = Period('3/12/12', freq='D')
        assert i1 == i2.asfreq('B')
        i3 = Period('3/10/12', freq='b')
        assert i1 == i3
        i1 = Period(year=2012, month=3, day=10, freq='B')
        i2 = Period('3/12/12', freq='B')
        assert i1 == i2