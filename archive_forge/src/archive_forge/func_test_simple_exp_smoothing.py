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
def test_simple_exp_smoothing(self):
    fit1 = SimpleExpSmoothing(self.oildata_oil, initialization_method='legacy-heuristic').fit(0.2, optimized=False)
    fit2 = SimpleExpSmoothing(self.oildata_oil, initialization_method='legacy-heuristic').fit(0.6, optimized=False)
    fit3 = SimpleExpSmoothing(self.oildata_oil, initialization_method='estimated').fit()
    assert_almost_equal(fit1.forecast(1), [484.802468], 4)
    assert_almost_equal(fit1.level, [446.6565229, 448.21987962, 449.7084985, 444.49324656, 446.84886283, 445.59670028, 441.54386424, 450.26498098, 461.4216172, 474.49569042, 482.45033014, 484.80246797], 4)
    assert_almost_equal(fit2.forecast(1), [501.837461], 4)
    assert_almost_equal(fit3.forecast(1), [496.493543], 4)
    assert_almost_equal(fit3.params['smoothing_level'], 0.891998, 4)
    assert_almost_equal(fit3.params['initial_level'], 447.47844, 3)