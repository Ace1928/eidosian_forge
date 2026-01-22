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
def test_circfuncs_uint8(self):
    x = np.array([150, 10], dtype='uint8')
    assert_equal(stats.circmean(x, high=180), 170.0)
    assert_allclose(stats.circvar(x, high=180), 0.2339555554617, rtol=1e-07)
    assert_allclose(stats.circstd(x, high=180), 20.91551378, rtol=1e-07)