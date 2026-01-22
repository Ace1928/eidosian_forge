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
@pytest.fixture(scope='module')
def ses():
    rs = np.random.RandomState(0)
    e = rs.standard_normal(1200)
    y = e.copy()
    for i in range(1, 1200):
        y[i] = y[i - 1] + e[i] - 0.2 * e[i - 1]
    y = y[200:]
    index = pd.date_range('2000-1-1', periods=y.shape[0], freq=MONTH_END)
    return pd.Series(y, index=index, name='y')