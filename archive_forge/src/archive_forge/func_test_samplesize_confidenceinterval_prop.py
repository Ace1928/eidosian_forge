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
def test_samplesize_confidenceinterval_prop():
    nobs = 20
    ci = smprop.proportion_confint(12, nobs, alpha=0.05, method='normal')
    res = smprop.samplesize_confint_proportion(12.0 / nobs, (ci[1] - ci[0]) / 2)
    assert_almost_equal(res, nobs, decimal=13)