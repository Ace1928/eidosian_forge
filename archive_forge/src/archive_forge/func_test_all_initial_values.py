from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_all_initial_values():
    fit1 = ExponentialSmoothing(aust, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='estimated').fit()
    lvl = np.round(fit1.params['initial_level'])
    trend = np.round(fit1.params['initial_trend'], 1)
    seas = np.round(fit1.params['initial_seasons'], 1)
    fit2 = ExponentialSmoothing(aust, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='known', initial_level=lvl, initial_trend=trend, initial_seasonal=seas).fit()
    assert_allclose(fit2.params['initial_level'], lvl)
    assert_allclose(fit2.params['initial_trend'], trend)
    assert_allclose(fit2.params['initial_seasons'], seas)