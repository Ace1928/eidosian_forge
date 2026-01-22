import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
def test_exact_basic(self):
    for n in range(1, 51):
        pmf1 = _get_wilcoxon_distr(n)
        pmf2 = _get_wilcoxon_distr2(n)
        assert_equal(n * (n + 1) / 2 + 1, len(pmf1))
        assert_equal(sum(pmf1), 1)
        assert_array_almost_equal(pmf1, pmf2)