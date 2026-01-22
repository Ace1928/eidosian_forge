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
def test_input_high_variance(self):
    x = np.array([3251637.22, 620695.44, 11642969.0, 2223468.22, 85307500.0, 16494389.89, 917215.88, 11642969.0, 2145773.87, 4962000.0, 620695.44, 651234.5, 1907876.71, 4053297.88, 3251637.22, 3259103.08, 9547969.0, 20631286.23, 12807072.08, 2383819.84, 90114500.0, 17209575.46, 12852969.0, 2414609.99, 2170368.23])
    xt_yeo, lam_yeo = stats.yeojohnson(x)
    xt_box, lam_box = stats.boxcox(x + 1)
    assert_allclose(xt_yeo, xt_box, rtol=1e-06)
    assert_allclose(lam_yeo, lam_box, rtol=1e-06)