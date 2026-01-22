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
def test_equivalence_2indep():
    alpha = 0.05
    count1, nobs1 = (7, 34)
    count2, nobs2 = (1, 34)
    count1v, nobs1v = ([7, 1], 34)
    count2v, nobs2v = ([1, 7], 34)
    methods_both = [('diff', 'agresti-caffo'), ('diff', 'score'), ('diff', 'wald'), ('ratio', 'log'), ('ratio', 'log-adjusted'), ('ratio', 'score'), ('odds-ratio', 'logit'), ('odds-ratio', 'logit-adjusted'), ('odds-ratio', 'logit-smoothed'), ('odds-ratio', 'score')]
    for co, method in methods_both:
        low, upp = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare=co, method=method, alpha=2 * alpha, correction=False)
        res = smprop.tost_proportions_2indep(count1, nobs1, count2, nobs2, low, upp * 1.05, compare=co, method=method, correction=False)
        assert_allclose(res.pvalue, alpha, atol=1e-10)
        res = smprop.tost_proportions_2indep(count1, nobs1, count2, nobs2, low * 0.95, upp, compare=co, method=method, correction=False)
        assert_allclose(res.pvalue, alpha, atol=1e-10)
        if method == 'logit-smoothed':
            return
        res1 = res
        res = smprop.tost_proportions_2indep(count1v, nobs1v, count2v, nobs2v, low * 0.95, upp, compare=co, method=method, correction=False)
        assert_allclose(res.pvalue[0], alpha, atol=1e-10)