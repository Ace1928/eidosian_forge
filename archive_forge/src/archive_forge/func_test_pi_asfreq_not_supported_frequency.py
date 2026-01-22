import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', ['2BMS', '2YS-MAR', '2bh'])
def test_pi_asfreq_not_supported_frequency(self, freq):
    msg = f'{freq[1:]} is not supported as period frequency'
    pi = PeriodIndex(['2020-01-01', '2021-01-01'], freq='M')
    with pytest.raises(ValueError, match=msg):
        pi.asfreq(freq=freq)