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
@pytest.mark.parametrize('n', [-1, 1, 3])
def test_construct_int_arg_no_kwargs_assumed_days(n):
    offset = DateOffset(n)
    assert offset._offset == timedelta(1)
    result = Timestamp(2022, 1, 2) + offset
    expected = Timestamp(2022, 1, 2 + n)
    assert result == expected