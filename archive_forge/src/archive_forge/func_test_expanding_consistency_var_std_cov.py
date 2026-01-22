import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('ddof', [0, 1])
def test_expanding_consistency_var_std_cov(all_data, min_periods, ddof):
    var_x = all_data.expanding(min_periods=min_periods).var(ddof=ddof)
    assert not (var_x < 0).any().any()
    std_x = all_data.expanding(min_periods=min_periods).std(ddof=ddof)
    assert not (std_x < 0).any().any()
    tm.assert_equal(var_x, std_x * std_x)
    cov_x_x = all_data.expanding(min_periods=min_periods).cov(all_data, ddof=ddof)
    assert not (cov_x_x < 0).any().any()
    tm.assert_equal(var_x, cov_x_x)