from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
@pytest.mark.parametrize('arithmatic_offset_type, n, expected', zip(_ARITHMETIC_DATE_OFFSET, range(1, 10), ['2009-01-02', '2008-03-02', '2008-01-23', '2008-01-06', '2008-01-02 05:00:00', '2008-01-02 00:06:00', '2008-01-02 00:00:07', '2008-01-02 00:00:00.008000000', '2008-01-02 00:00:00.000009000']))
def test_mul_add(self, arithmatic_offset_type, n, expected, dt):
    assert DateOffset(**{arithmatic_offset_type: 1}) * n + dt == Timestamp(expected)
    assert n * DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
    assert dt + DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
    assert dt + n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)