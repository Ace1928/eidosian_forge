from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('freq, freq_depr', [('2ME', '2M'), ('2SME', '2SM'), ('2SME', '2sm'), ('2QE', '2Q'), ('2QE-SEP', '2Q-SEP'), ('1YE', '1Y'), ('2YE-MAR', '2Y-MAR'), ('1YE', '1A'), ('2YE-MAR', '2A-MAR'), ('2ME', '2m'), ('2QE-SEP', '2q-sep'), ('2YE-MAR', '2a-mar'), ('2YE', '2y')])
def test_date_range_frequency_M_Q_Y_A_deprecated(self, freq, freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."
    expected = pd.date_range('1/1/2000', periods=4, freq=freq)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = pd.date_range('1/1/2000', periods=4, freq=freq_depr)
    tm.assert_index_equal(result, expected)