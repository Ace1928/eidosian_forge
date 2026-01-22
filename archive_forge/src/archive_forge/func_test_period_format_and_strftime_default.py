from contextlib import nullcontext
from datetime import (
import locale
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_period_format_and_strftime_default(self):
    per = PeriodIndex([datetime(2003, 1, 1, 12), None], freq='h')
    msg = 'PeriodIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = per.format()
    assert formatted[0] == '2003-01-01 12:00'
    assert formatted[1] == 'NaT'
    assert formatted[0] == per.strftime(None)[0]
    assert per.strftime(None)[1] is np.nan
    per = pd.period_range('2003-01-01 12:01:01.123456789', periods=2, freq='ns')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = per.format()
    assert (formatted == per.strftime(None)).all()
    assert formatted[0] == '2003-01-01 12:01:01.123456789'
    assert formatted[1] == '2003-01-01 12:01:01.123456790'