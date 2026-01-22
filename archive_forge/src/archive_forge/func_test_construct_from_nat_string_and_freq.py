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
@pytest.mark.parametrize('freq', ['Y', 'M', 'D', 'h'])
def test_construct_from_nat_string_and_freq(self, freq):
    per = Period('NaT', freq=freq)
    assert per is NaT
    per = Period('NaT', freq='2' + freq)
    assert per is NaT
    per = Period('NaT', freq='3' + freq)
    assert per is NaT