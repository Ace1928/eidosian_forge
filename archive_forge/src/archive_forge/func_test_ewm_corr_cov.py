import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name', ['cov', 'corr'])
def test_ewm_corr_cov(name):
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)
    A[:10] = np.nan
    B.iloc[-10:] = np.nan
    result = getattr(A.ewm(com=20, min_periods=5), name)(B)
    assert np.isnan(result.values[:14]).all()
    assert not np.isnan(result.values[14:]).any()