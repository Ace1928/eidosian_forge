from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_range_error():
    idx = pd.Index([0, 1, 1, 2, 3, 5, 8, 13])
    dp = DeterministicProcess(idx, constant=True, order=2, seasonal=True, period=2)
    with pytest.raises(TypeError, match='The index in the deterministic'):
        dp.range(0, 12)