from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_invalid_freq_period(time_index):
    with pytest.raises(ValueError, match='The combination of freq='):
        CalendarSeasonality('h', YEAR_END)
    cs = CalendarSeasonality('B', 'W')
    with pytest.raises(ValueError, match='freq is B but index contains'):
        cs.in_sample(pd.date_range('2000-1-1', periods=10, freq='D'))