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
def test_2_args_ttest(self):
    res_tukey = stats.tukey_hsd(*self.data_diff_size[:2])
    res_ttest = stats.ttest_ind(*self.data_diff_size[:2])
    assert_allclose(res_ttest.pvalue, res_tukey.pvalue[0, 1])
    assert_allclose(res_ttest.pvalue, res_tukey.pvalue[1, 0])