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
def test_to_timestamp_mult(self):
    p = Period('2011-01', freq='M')
    assert p.to_timestamp(how='S') == Timestamp('2011-01-01')
    expected = Timestamp('2011-02-01') - Timedelta(1, 'ns')
    assert p.to_timestamp(how='E') == expected
    p = Period('2011-01', freq='3M')
    assert p.to_timestamp(how='S') == Timestamp('2011-01-01')
    expected = Timestamp('2011-04-01') - Timedelta(1, 'ns')
    assert p.to_timestamp(how='E') == expected