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
@pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
@pytest.mark.parametrize('equal_var', [False, True])
@pytest.mark.parametrize('trim', [0, 0.2])
def test_confidence_interval(self, alternative, equal_var, trim):
    if equal_var and trim:
        pytest.xfail('Discrepancy in `main`; needs further investigation.')
    rng = np.random.default_rng(3810954496107292580)
    x = rng.random(11)
    y = rng.random(13)
    res = stats.ttest_ind(x, y, alternative=alternative, equal_var=equal_var, trim=trim)
    alternatives = {'two-sided': 0, 'less': 1, 'greater': 2}
    ref = self.r[alternatives[alternative], int(equal_var), int(np.ceil(trim))]
    statistic, df, pvalue, low, high = ref
    assert_allclose(res.statistic, statistic)
    assert_allclose(res.df, df)
    assert_allclose(res.pvalue, pvalue)
    if not equal_var:
        ci = res.confidence_interval(0.9)
        assert_allclose(ci.low, low)
        assert_allclose(ci.high, high)