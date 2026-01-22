from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.smoke
def test_calendar_time_trend_smoke(time_index, forecast_index):
    ct = CalendarTimeTrend(YEAR_END, order=2)
    ct.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    ct.out_of_sample(steps, time_index, forecast_index)
    str(ct)
    hash(ct)
    assert isinstance(ct.order, int)
    assert isinstance(ct.constant, bool)
    assert isinstance(ct.freq, str)
    assert ct.base_period is None