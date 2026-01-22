import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
def test_ztost():
    xfair = np.repeat([1, 0], [228, 762 - 228])
    from statsmodels.stats.weightstats import zconfint, ztost
    ci01 = zconfint(xfair, alpha=0.1, ddof=0)
    assert_almost_equal(ci01, [0.2719, 0.3265], 4)
    res = ztost(xfair, 0.18, 0.38, ddof=0)
    assert_almost_equal(res[1][0], 7.1865, 4)
    assert_almost_equal(res[2][0], -4.8701, 4)
    assert_array_less(res[0], 0.0001)