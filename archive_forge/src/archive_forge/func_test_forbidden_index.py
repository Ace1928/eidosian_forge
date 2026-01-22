from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_forbidden_index():
    index = pd.RangeIndex(0, 10)
    ct = CalendarTimeTrend(YEAR_END, order=2)
    with pytest.raises(TypeError, match='CalendarTimeTrend terms can only'):
        ct.in_sample(index)