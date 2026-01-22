from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
@pytest.fixture
def statespace_comparison(austourists):
    ets_model = ETSModel(austourists, seasonal_periods=4, error='add', trend='add', seasonal='add', damped_trend=True)
    ets_results = ets_model.fit(disp=False)
    statespace_model = statespace.ExponentialSmoothing(austourists, trend=True, damped_trend=True, seasonal=4, initialization_method='known', initial_level=ets_results.initial_level, initial_trend=ets_results.initial_trend, initial_seasonal=ets_results.initial_seasonal[::-1])
    with statespace_model.fix_params({'smoothing_level': ets_results.smoothing_level, 'smoothing_trend': ets_results.smoothing_trend, 'smoothing_seasonal': ets_results.smoothing_seasonal, 'damping_trend': ets_results.damping_trend}):
        statespace_results = statespace_model.fit()
    ets_results.test_serial_correlation('ljungbox')
    statespace_results.test_serial_correlation('ljungbox')
    return (ets_results, statespace_results)