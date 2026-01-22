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
def test_hw_seasonal_buggy(self):
    fit3 = ExponentialSmoothing(self.aust, seasonal_periods=4, seasonal='add', initialization_method='estimated', use_boxcox=True).fit()
    assert_almost_equal(fit3.forecast(8), [59.48719, 35.758854, 44.600641, 47.751384, 59.48719, 35.758854, 44.600641, 47.751384], 2)
    fit4 = ExponentialSmoothing(self.aust, seasonal_periods=4, seasonal='mul', initialization_method='estimated', use_boxcox=True).fit()
    assert_almost_equal(fit4.forecast(8), [59.26155037, 35.27811302, 44.00438543, 47.97732693, 59.26155037, 35.27811302, 44.00438543, 47.97732693], 2)