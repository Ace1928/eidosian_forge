from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_seasonal_from_index_err():
    index = pd.Index([0, 1, 1, 2, 3, 5, 8, 12])
    with pytest.raises(TypeError):
        Seasonality.from_index(index)
    index = pd.date_range('2000-1-1', periods=10)[[0, 1, 2, 3, 5, 8]]
    with pytest.raises(ValueError):
        Seasonality.from_index(index)