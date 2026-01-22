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
def test_example1a(self):
    t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
    t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
    t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
    t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])
    Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=False)
    assert_almost_equal(Tk, 4.449, 3)
    assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.493, 3.2459], tm[0:5], 4)
    assert_allclose(p, 0.0021, atol=0.00025)