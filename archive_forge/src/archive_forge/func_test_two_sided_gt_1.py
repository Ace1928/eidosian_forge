from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_two_sided_gt_1(self):
    tbl = [[1, 1], [13, 12]]
    pl = boschloo_exact(tbl, alternative='less').pvalue
    pg = boschloo_exact(tbl, alternative='greater').pvalue
    assert 2 * min(pl, pg) > 1
    pt = boschloo_exact(tbl, alternative='two-sided').pvalue
    assert pt == 1.0