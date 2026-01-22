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
def test_compare_r(self):
    """
        Testing against results and p-values from R:
        from: https://www.rdocumentation.org/packages/stats/versions/3.6.2/
        topics/TukeyHSD
        > require(graphics)
        > summary(fm1 <- aov(breaks ~ tension, data = warpbreaks))
        > TukeyHSD(fm1, "tension", ordered = TRUE)
        > plot(TukeyHSD(fm1, "tension"))
        Tukey multiple comparisons of means
        95% family-wise confidence level
        factor levels have been ordered
        Fit: aov(formula = breaks ~ tension, data = warpbreaks)
        $tension
        """
    str_res = '\n                diff        lwr      upr     p adj\n        2 - 3  4.722222 -4.8376022 14.28205 0.4630831\n        1 - 3 14.722222  5.1623978 24.28205 0.0014315\n        1 - 2 10.000000  0.4401756 19.55982 0.0384598\n        '
    res_expect = np.asarray(str_res.replace(' - ', ' ').split()[5:], dtype=float).reshape((3, 6))
    data = ([26, 30, 54, 25, 70, 52, 51, 26, 67, 27, 14, 29, 19, 29, 31, 41, 20, 44], [18, 21, 29, 17, 12, 18, 35, 30, 36, 42, 26, 19, 16, 39, 28, 21, 39, 29], [36, 21, 24, 18, 10, 43, 28, 15, 26, 20, 21, 24, 17, 13, 15, 15, 16, 28])
    res_tukey = stats.tukey_hsd(*data)
    conf = res_tukey.confidence_interval()
    for i, j, s, l, h, p in res_expect:
        i, j = (int(i) - 1, int(j) - 1)
        assert_allclose(conf.low[i, j], l, atol=1e-07)
        assert_allclose(res_tukey.statistic[i, j], s, atol=1e-06)
        assert_allclose(conf.high[i, j], h, atol=1e-05)
        assert_allclose(res_tukey.pvalue[i, j], p, atol=1e-07)