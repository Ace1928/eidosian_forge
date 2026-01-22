from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def test_roll_date_object(self):
    offset = CBMonthEnd()
    dt = date(2012, 9, 15)
    result = offset.rollback(dt)
    assert result == datetime(2012, 8, 31)
    result = offset.rollforward(dt)
    assert result == datetime(2012, 9, 28)
    offset = offsets.Day()
    result = offset.rollback(dt)
    assert result == datetime(2012, 9, 15)
    result = offset.rollforward(dt)
    assert result == datetime(2012, 9, 15)