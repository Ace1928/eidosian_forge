from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_seasonality_time_index(time_index):
    tt = Seasonality.from_index(time_index)
    assert tt.period == 5
    fcast = tt.out_of_sample(steps=12, index=time_index)
    new_idx = DeterministicTerm._extend_index(time_index, 12)
    pd.testing.assert_index_equal(fcast.index, new_idx)