from __future__ import annotations
from datetime import datetime
import pytest
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_bad_month_fail(self):
    msg = 'Month must go from 1 to 12'
    with pytest.raises(ValueError, match=msg):
        BYearEnd(month=13)
    with pytest.raises(ValueError, match=msg):
        BYearEnd(month=0)