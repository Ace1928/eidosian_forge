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
@pytest.mark.parametrize('method', ['least_squares', 'basinhopping', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr'])
def test_alternative_minimizers(method, ses):
    sv = np.array([0.77, 11.0])
    minimize_kwargs = {}
    mod = ExponentialSmoothing(ses, initialization_method='estimated')
    res = mod.fit(method=method, start_params=sv, minimize_kwargs=minimize_kwargs)
    assert_allclose(res.params['smoothing_level'], 0.77232545, rtol=0.001)
    assert_allclose(res.params['initial_level'], 11.00359693, rtol=0.001)
    assert isinstance(res.summary().as_text(), str)