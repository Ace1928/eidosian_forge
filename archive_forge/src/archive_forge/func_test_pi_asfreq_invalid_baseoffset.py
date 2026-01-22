import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', [offsets.MonthBegin(2), offsets.BusinessMonthEnd(2)])
def test_pi_asfreq_invalid_baseoffset(self, freq):
    msg = re.escape(f'{freq} is not supported as period frequency')
    pi = PeriodIndex(['2020-01-01', '2021-01-01'], freq='M')
    with pytest.raises(ValueError, match=msg):
        pi.asfreq(freq=freq)