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
@pytest.mark.parametrize('alternative', ('two-sided', 'less', 'greater'))
def test_alternative_nan_policy(self, alternative):
    x1 = [1, 2, 3, 4, 5]
    x2 = [5, 6, 7, 8, 7]
    x1nan = x1 + [np.nan]
    x2nan = x2 + [np.nan]
    assert_array_equal(stats.spearmanr(x1nan, x2nan), (np.nan, np.nan))
    res_actual = stats.spearmanr(x1nan, x2nan, nan_policy='omit', alternative=alternative)
    res_expected = stats.spearmanr(x1, x2, alternative=alternative)
    assert_allclose(res_actual, res_expected)
    message = 'The input contains nan values'
    with pytest.raises(ValueError, match=message):
        stats.spearmanr(x1nan, x2nan, nan_policy='raise', alternative=alternative)
    message = 'nan_policy must be one of...'
    with pytest.raises(ValueError, match=message):
        stats.spearmanr(x1nan, x2nan, nan_policy='ekki-ekki', alternative=alternative)