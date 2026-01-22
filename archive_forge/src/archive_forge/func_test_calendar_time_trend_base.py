from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_calendar_time_trend_base(time_index):
    ct = CalendarTimeTrend(MONTH_END, True, order=3, base_period='1960-1-1')
    ct2 = CalendarTimeTrend(MONTH_END, True, order=3)
    assert ct != ct2
    str(ct)
    str(ct2)
    assert ct.base_period is not None
    assert ct2.base_period is None