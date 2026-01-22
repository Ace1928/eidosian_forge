from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.parametrize('freq_period', cs_params)
def test_calendar_seasonality(time_index, forecast_index, freq_period):
    freq, period = freq_period
    cs = CalendarSeasonality(period, freq)
    cs.in_sample(time_index)
    steps = 83 if forecast_index is None else len(forecast_index)
    cs.out_of_sample(steps, time_index, forecast_index)
    assert isinstance(cs.period, str)
    assert isinstance(cs.freq, str)
    str(cs)
    repr(cs)
    hash(cs)
    cs2 = CalendarSeasonality(period, freq)
    assert cs == cs2