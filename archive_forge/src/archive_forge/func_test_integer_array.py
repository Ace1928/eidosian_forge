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
def test_integer_array(reset_randomstate):
    rs = np.random.RandomState(12345)
    e = 10 * rs.standard_normal((1000, 2))
    y_star = np.cumsum(e[:, 0])
    y = y_star + e[:, 1]
    y = y.astype(int)
    res = ExponentialSmoothing(y, trend='add', initialization_method='estimated').fit()
    assert res.params['smoothing_level'] != 0.0