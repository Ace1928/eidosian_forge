import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_normal_sample_size_one_tail():
    smp.normal_sample_size_one_tail(5, 0.8, 0.05, 2, std_alternative=None)
    alphas = np.asarray([0.01, 0.05, 0.1, 0.5, 0.8])
    powers = np.asarray([0.99, 0.95, 0.9, 0.5, 0.2])
    nobs_with_zeros = smp.normal_sample_size_one_tail(5, powers, alphas, 2, 2)
    assert_array_equal(nobs_with_zeros[powers <= alphas], 0)