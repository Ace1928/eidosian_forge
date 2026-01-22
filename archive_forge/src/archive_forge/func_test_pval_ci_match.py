import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
@pytest.mark.parametrize('case', ['continuous', 'discrete'])
@pytest.mark.parametrize('alternative', ['less', 'greater'])
@pytest.mark.parametrize('alpha', [0.9, 0.95])
def test_pval_ci_match(self, case, alternative, alpha):
    seed = int((7 ** len(case) + len(alternative)) * alpha)
    rng = np.random.default_rng(seed)
    if case == 'continuous':
        p, q = rng.random(size=2)
        rvs = rng.random(size=100)
    else:
        rvs = rng.integers(1, 11, size=100)
        p = rng.random()
        q = rng.integers(1, 11)
    res = stats.quantile_test(rvs, q=q, p=p, alternative=alternative)
    ci = res.confidence_interval(confidence_level=alpha)
    if alternative == 'less':
        i_inside = rvs <= ci.high
    else:
        i_inside = rvs >= ci.low
    for x in rvs[i_inside]:
        res = stats.quantile_test(rvs, q=x, p=p, alternative=alternative)
        assert res.pvalue > 1 - alpha
    for x in rvs[~i_inside]:
        res = stats.quantile_test(rvs, q=x, p=p, alternative=alternative)
        assert res.pvalue < 1 - alpha