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
@pytest.mark.parametrize('method', ['estimated', 'heuristic', 'legacy-heuristic'])
@pytest.mark.parametrize('trend', [None, 'add'])
@pytest.mark.parametrize('seasonal', [None, 'add'])
def test_initialization_methods(ses, method, trend, seasonal):
    mod = ExponentialSmoothing(ses, trend=trend, seasonal=seasonal, initialization_method=method)
    res = mod.fit()
    assert res.mle_retvals.success
    assert isinstance(res.summary().as_text(), str)