from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.parametrize('constant', [True, False])
@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('seasonal', [True, False])
@pytest.mark.parametrize('fourier', [0, 1])
@pytest.mark.parametrize('period', [None, 10])
@pytest.mark.parametrize('drop', [True, False])
def test_deterministic_process(time_index, constant, order, seasonal, fourier, period, drop):
    if seasonal and fourier:
        return
    dp = DeterministicProcess(time_index, constant=constant, order=order, seasonal=seasonal, fourier=fourier, period=period, drop=drop)
    terms = dp.in_sample()
    pd.testing.assert_index_equal(terms.index, time_index)
    terms = dp.out_of_sample(23)
    assert isinstance(terms, pd.DataFrame)