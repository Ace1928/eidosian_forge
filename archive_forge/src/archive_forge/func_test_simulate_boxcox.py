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
def test_simulate_boxcox(austourists):
    """
    check if simulation results with boxcox fits are reasonable
    """
    fit = ExponentialSmoothing(austourists, seasonal_periods=4, trend='add', seasonal='mul', damped_trend=False, initialization_method='estimated', use_boxcox=True).fit()
    expected = fit.forecast(4).values
    res = fit.simulate(4, repetitions=10, random_state=0).values
    mean = np.mean(res, axis=1)
    assert np.all(np.abs(mean - expected) < 5)