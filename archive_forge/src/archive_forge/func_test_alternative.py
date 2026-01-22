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
def test_alternative(self):
    x1 = [1, 2, 3, 4, 5]
    x2 = [5, 6, 7, 8, 7]
    expected = (0.8207826816681233, 0.0885870053135438)
    res = stats.spearmanr(x1, x2, alternative='less')
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], 1 - expected[1] / 2)
    res = stats.spearmanr(x1, x2, alternative='greater')
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], expected[1] / 2)
    with pytest.raises(ValueError, match="alternative must be 'less'..."):
        stats.spearmanr(x1, x2, alternative='ekki-ekki')