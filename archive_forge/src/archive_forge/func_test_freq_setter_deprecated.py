import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
def test_freq_setter_deprecated(self):
    idx = period_range('2018Q1', periods=4, freq='Q')
    with tm.assert_produces_warning(None):
        idx.freq
    msg = "property 'freq' of 'PeriodArray' object has no setter" if PY311 else "can't set attribute"
    with pytest.raises(AttributeError, match=msg):
        idx.freq = offsets.Day()