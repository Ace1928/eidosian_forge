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
def test_ttest_single_observation():
    rng = np.random.default_rng(246834602926842)
    x = rng.normal(size=(10000, 2))
    y = rng.normal(size=(10000, 1))
    q = rng.uniform(size=100)
    res = stats.ttest_ind(x, y, equal_var=True, axis=-1)
    assert stats.ks_1samp(res.pvalue, stats.uniform().cdf).pvalue > 0.1
    assert_allclose(np.percentile(res.pvalue, q * 100), q, atol=0.01)
    res = stats.ttest_ind(y, x, equal_var=True, axis=-1)
    assert stats.ks_1samp(res.pvalue, stats.uniform().cdf).pvalue > 0.1
    assert_allclose(np.percentile(res.pvalue, q * 100), q, atol=0.01)
    res = stats.ttest_ind([2, 3, 5], [1.5], equal_var=True)
    assert_allclose(res, (1.0394023007754, 0.407779907736), rtol=1e-10)