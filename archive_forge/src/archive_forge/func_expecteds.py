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
@pytest.fixture
def expecteds():
    return {'Day': Timestamp('2011-01-02 09:00:00'), 'DateOffset': Timestamp('2011-01-02 09:00:00'), 'BusinessDay': Timestamp('2011-01-03 09:00:00'), 'CustomBusinessDay': Timestamp('2011-01-03 09:00:00'), 'CustomBusinessMonthEnd': Timestamp('2011-01-31 09:00:00'), 'CustomBusinessMonthBegin': Timestamp('2011-01-03 09:00:00'), 'MonthBegin': Timestamp('2011-02-01 09:00:00'), 'BusinessMonthBegin': Timestamp('2011-01-03 09:00:00'), 'MonthEnd': Timestamp('2011-01-31 09:00:00'), 'SemiMonthEnd': Timestamp('2011-01-15 09:00:00'), 'SemiMonthBegin': Timestamp('2011-01-15 09:00:00'), 'BusinessMonthEnd': Timestamp('2011-01-31 09:00:00'), 'YearBegin': Timestamp('2012-01-01 09:00:00'), 'BYearBegin': Timestamp('2011-01-03 09:00:00'), 'YearEnd': Timestamp('2011-12-31 09:00:00'), 'BYearEnd': Timestamp('2011-12-30 09:00:00'), 'QuarterBegin': Timestamp('2011-03-01 09:00:00'), 'BQuarterBegin': Timestamp('2011-03-01 09:00:00'), 'QuarterEnd': Timestamp('2011-03-31 09:00:00'), 'BQuarterEnd': Timestamp('2011-03-31 09:00:00'), 'BusinessHour': Timestamp('2011-01-03 10:00:00'), 'CustomBusinessHour': Timestamp('2011-01-03 10:00:00'), 'WeekOfMonth': Timestamp('2011-01-08 09:00:00'), 'LastWeekOfMonth': Timestamp('2011-01-29 09:00:00'), 'FY5253Quarter': Timestamp('2011-01-25 09:00:00'), 'FY5253': Timestamp('2011-01-25 09:00:00'), 'Week': Timestamp('2011-01-08 09:00:00'), 'Easter': Timestamp('2011-04-24 09:00:00'), 'Hour': Timestamp('2011-01-01 10:00:00'), 'Minute': Timestamp('2011-01-01 09:01:00'), 'Second': Timestamp('2011-01-01 09:00:01'), 'Milli': Timestamp('2011-01-01 09:00:00.001000'), 'Micro': Timestamp('2011-01-01 09:00:00.000001'), 'Nano': Timestamp('2011-01-01T09:00:00.000000001')}