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
@pytest.mark.parametrize('trend', TRENDS)
@pytest.mark.parametrize('seasonal', SEASONALS)
@pytest.mark.parametrize('damped', (True, False))
@pytest.mark.parametrize('error', ('add', 'mul'))
def test_simulate_expected_r(trend, seasonal, damped, error, austourists, simulate_expected_results_r, simulate_fit_state_r):
    """
    Test for :meth:``statsmodels.tsa.holtwinters.HoltWintersResults``.

    The tests are using the implementation in the R package ``forecast`` as
    reference, and example data is taken from ``fpp2`` (package and book).
    """
    short_name = {'add': 'A', 'mul': 'M', None: 'N'}
    model_name = short_name[error] + short_name[trend] + short_name[seasonal]
    if model_name in simulate_expected_results_r[damped]:
        expected = np.asarray(simulate_expected_results_r[damped][model_name])
        state = simulate_fit_state_r[damped][model_name]
    else:
        return
    fit = ExponentialSmoothing(austourists, seasonal_periods=4, trend=trend, seasonal=seasonal, damped_trend=damped).fit(smoothing_level=state['alpha'], smoothing_trend=state['beta'], smoothing_seasonal=state['gamma'], damping_trend=state['phi'], optimized=False)
    fit._level[-1] = state['l']
    fit._trend[-1] = state['b']
    fit._season[-1] = state['s1']
    fit._season[-2] = state['s2']
    fit._season[-3] = state['s3']
    fit._season[-4] = state['s4']
    if np.any(np.isnan(fit.fittedvalues)):
        return
    innov = np.asarray([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]).T
    sim = fit.simulate(4, repetitions=1, error=error, random_errors=innov)
    assert_almost_equal(expected, sim.values, 5)