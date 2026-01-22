import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bias', [True, False])
def test_ewm_consistency_series_cov_corr(series_data, adjust, ignore_na, min_periods, bias):
    com = 3.0
    var_x_plus_y = (series_data + series_data).ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).var(bias=bias)
    var_x = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).var(bias=bias)
    var_y = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).var(bias=bias)
    cov_x_y = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).cov(series_data, bias=bias)
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))
    corr_x_y = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).corr(series_data)
    std_x = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).std(bias=bias)
    std_y = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).std(bias=bias)
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))
    if bias:
        mean_x = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).mean()
        mean_y = series_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).mean()
        mean_x_times_y = (series_data * series_data).ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).mean()
        tm.assert_equal(cov_x_y, mean_x_times_y - mean_x * mean_y)