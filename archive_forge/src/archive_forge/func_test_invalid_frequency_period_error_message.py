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
def test_invalid_frequency_period_error_message(self):
    msg = "for Period, please use 'M' instead of 'ME'"
    with pytest.raises(ValueError, match=msg):
        Period('2012-01-02', freq='ME')