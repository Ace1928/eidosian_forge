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
@pytest.mark.parametrize('data,res_expect_str,atol', ((data_same_size, matlab_sm_siz, 1e-12), (data_diff_size, matlab_diff_sz, 1e-07)), ids=['equal size sample', 'unequal size sample'])
def test_compare_matlab(self, data, res_expect_str, atol):
    """
        vals = [24.5, 23.5,  26.4, 27.1, 29.9, 28.4, 34.2, 29.5, 32.2, 30.1,
         26.1, 28.3, 24.3, 26.2, 27.8]
        names = {'zero', 'zero', 'zero', 'zero', 'zero', 'one', 'one', 'one',
         'one', 'one', 'two', 'two', 'two', 'two', 'two'}
        [p,t,stats] = anova1(vals,names,"off");
        [c,m,h,nms] = multcompare(stats, "CType","hsd");
        """
    res_expect = np.asarray(res_expect_str.split(), dtype=float).reshape((3, 6))
    res_tukey = stats.tukey_hsd(*data)
    conf = res_tukey.confidence_interval()
    for i, j, l, s, h, p in res_expect:
        i, j = (int(i) - 1, int(j) - 1)
        assert_allclose(conf.low[i, j], l, atol=atol)
        assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
        assert_allclose(conf.high[i, j], h, atol=atol)
        assert_allclose(res_tukey.pvalue[i, j], p, atol=atol)