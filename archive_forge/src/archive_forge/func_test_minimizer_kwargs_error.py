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
def test_minimizer_kwargs_error(ses):
    mod = ExponentialSmoothing(ses, initialization_method='estimated')
    kwargs = {'args': 'anything'}
    with pytest.raises(ValueError):
        mod.fit(minimize_kwargs=kwargs)
    with pytest.raises(ValueError):
        mod.fit(method='least_squares', minimize_kwargs=kwargs)
    kwargs = {'minimizer_kwargs': {'args': 'anything'}}
    with pytest.raises(ValueError):
        mod.fit(method='basinhopping', minimize_kwargs=kwargs)
    kwargs = {'minimizer_kwargs': {'method': 'SLSQP'}}
    res = mod.fit(method='basinhopping', minimize_kwargs=kwargs)
    assert isinstance(res.params, dict)
    assert isinstance(res.summary().as_text(), str)