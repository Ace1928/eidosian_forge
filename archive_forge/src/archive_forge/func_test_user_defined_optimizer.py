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
def test_user_defined_optimizer(self):
    lmbda = stats.boxcox_normmax(self.x)
    lmbda_rounded = np.round(lmbda, 5)
    lmbda_range = np.linspace(lmbda_rounded - 0.01, lmbda_rounded + 0.01, 1001)

    class MyResult:
        pass

    def optimizer(fun):
        objs = []
        for lmbda in lmbda_range:
            objs.append(fun(lmbda))
        res = MyResult()
        res.x = lmbda_range[np.argmin(objs)]
        return res
    lmbda2 = stats.boxcox_normmax(self.x, optimizer=optimizer)
    assert lmbda2 != lmbda
    assert_allclose(lmbda2, lmbda, 1e-05)