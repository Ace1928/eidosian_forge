from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.tools.datetimes import to_datetime
from pandas.tseries import (
@pytest.mark.parametrize('expected,dates', list({'YS-JAN': ['2009-01-01', '2010-01-01', '2011-01-01', '2012-01-01'], 'QE-OCT': ['2009-01-31', '2009-04-30', '2009-07-31', '2009-10-31'], 'ME': ['2010-11-30', '2010-12-31', '2011-01-31', '2011-02-28'], 'W-SAT': ['2010-12-25', '2011-01-01', '2011-01-08', '2011-01-15'], 'D': ['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04'], 'h': ['2011-12-31 22:00', '2011-12-31 23:00', '2012-01-01 00:00', '2012-01-01 01:00']}.items()))
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_infer_freq_tz(tz_naive_fixture, expected, dates, unit):
    tz = tz_naive_fixture
    idx = DatetimeIndex(dates, tz=tz).as_unit(unit)
    assert idx.inferred_freq == expected