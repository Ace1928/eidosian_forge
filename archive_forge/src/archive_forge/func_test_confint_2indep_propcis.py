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
def test_confint_2indep_propcis():
    count1, nobs1 = (7, 34)
    count2, nobs2 = (1, 34)
    ci = (0.0270416, 0.3452912)
    ci1 = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='diff', method='score', correction=True)
    assert_allclose(ci1, ci, atol=0.002)
    ci = (0.01161167, 0.32172166)
    ci1 = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='diff', method='agresti-caffo')
    assert_allclose(ci1, ci, atol=6e-07)
    ci = (0.02916942, 0.32377176)
    ci1 = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='diff', method='wald', correction=False)
    assert_allclose(ci1, ci, atol=6e-07)
    ci = (1.246309, 56.48613)
    ci1 = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='odds-ratio', method='score', correction=True)
    assert_allclose(ci1, ci, rtol=0.0005)
    ci = (1.220853, 42.575718)
    ci1 = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare='ratio', method='score', correction=False)
    assert_allclose(ci1, ci, atol=6e-07)