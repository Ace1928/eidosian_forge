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
def test_example_conover(self):
    x = [7.6, 8.4, 8.6, 8.7, 9.3, 9.9, 10.1, 10.6, 11.2]
    y = [5.2, 5.7, 5.9, 6.5, 6.8, 8.2, 9.1, 9.8, 10.8, 11.3, 11.5, 12.3, 12.5, 13.4, 14.6]
    r = cramervonmises_2samp(x, y)
    assert_allclose(r.statistic, 0.262, atol=0.001)
    assert_allclose(r.pvalue, 0.18, atol=0.01)