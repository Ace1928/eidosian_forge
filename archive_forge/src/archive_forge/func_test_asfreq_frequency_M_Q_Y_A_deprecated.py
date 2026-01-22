from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq, freq_depr', [('2ME', '2M'), ('2QE', '2Q'), ('2QE-SEP', '2Q-SEP'), ('1BQE', '1BQ'), ('2BQE-SEP', '2BQ-SEP'), ('1YE', '1Y'), ('2YE-MAR', '2Y-MAR'), ('1YE', '1A'), ('2YE-MAR', '2A-MAR'), ('2BYE-MAR', '2BA-MAR')])
def test_asfreq_frequency_M_Q_Y_A_deprecated(self, freq, freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."
    index = date_range('1/1/2000', periods=4, freq=f'{freq[1:]}')
    df = DataFrame({'s': Series([0.0, 1.0, 2.0, 3.0], index=index)})
    expected = df.asfreq(freq=freq)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = df.asfreq(freq=freq_depr)
    tm.assert_frame_equal(result, expected)