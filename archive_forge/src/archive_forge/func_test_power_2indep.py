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
def test_power_2indep():
    pow_ = power_proportions_2indep(-0.25, 0.75, 76.70692)
    assert_allclose(pow_.power, 0.9, atol=1e-08)
    n = samplesize_proportions_2indep_onetail(-0.25, 0.75, 0.9, ratio=1, alpha=0.05, value=0, alternative='two-sided')
    assert_allclose(n, 76.70692, atol=1e-05)
    power_proportions_2indep(-0.25, 0.75, 62.33551, alternative='smaller')
    assert_allclose(pow_.power, 0.9, atol=1e-08)
    pow_ = power_proportions_2indep(0.25, 0.5, 62.33551, alternative='smaller')
    assert_array_less(pow_.power, 0.05)
    pow_ = power_proportions_2indep(0.25, 0.5, 62.33551, alternative='larger', return_results=False)
    assert_allclose(pow_, 0.9, atol=1e-08)
    pow_ = power_proportions_2indep(-0.15, 0.65, 83.4373, return_results=False)
    assert_allclose(pow_, 0.5, atol=1e-08)
    n = samplesize_proportions_2indep_onetail(-0.15, 0.65, 0.5, ratio=1, alpha=0.05, value=0, alternative='two-sided')
    assert_allclose(n, 83.4373, atol=0.05)
    from statsmodels.stats.power import normal_sample_size_one_tail
    res = power_proportions_2indep(-0.014, 0.015, 550, ratio=1.0)
    assert_allclose(res.power, 0.74156, atol=1e-07)
    n = normal_sample_size_one_tail(-0.014, 0.74156, 0.05 / 2, std_null=res.std_null, std_alternative=res.std_alt)
    assert_allclose(n, 550, atol=0.05)
    n2 = samplesize_proportions_2indep_onetail(-0.014, 0.015, 0.74156, ratio=1, alpha=0.05, value=0, alternative='two-sided')
    assert_allclose(n2, n, rtol=1e-13)
    pwr_st = 0.7995659211532175
    n = 154
    res = power_proportions_2indep(-0.1, 0.2, n, ratio=2.0)
    assert_allclose(res.power, pwr_st, atol=1e-07)
    n2 = samplesize_proportions_2indep_onetail(-0.1, 0.2, pwr_st, ratio=2)
    assert_allclose(n2, n, rtol=0.0001)