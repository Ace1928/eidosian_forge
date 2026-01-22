from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq, freq_half', [('2ME', 'ME'), (MonthEnd(2), MonthEnd(1))])
def test_asfreq_2ME(self, freq, freq_half):
    index = date_range('1/1/2000', periods=6, freq=freq_half)
    df = DataFrame({'s': Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=index)})
    expected = df.asfreq(freq=freq)
    index = date_range('1/1/2000', periods=3, freq=freq)
    result = DataFrame({'s': Series([0.0, 2.0, 4.0], index=index)})
    tm.assert_frame_equal(result, expected)