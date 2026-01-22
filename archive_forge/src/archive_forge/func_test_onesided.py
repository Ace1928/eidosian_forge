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
def test_onesided(self):
    x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
    y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='Sample size too small')
        w, p = stats.wilcoxon(x, y, alternative='less', mode='approx')
    assert_equal(w, 27)
    assert_almost_equal(p, 0.7031847, decimal=6)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='Sample size too small')
        w, p = stats.wilcoxon(x, y, alternative='less', correction=True, mode='approx')
    assert_equal(w, 27)
    assert_almost_equal(p, 0.7233656, decimal=6)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='Sample size too small')
        w, p = stats.wilcoxon(x, y, alternative='greater', mode='approx')
    assert_equal(w, 27)
    assert_almost_equal(p, 0.2968153, decimal=6)
    with suppress_warnings() as sup:
        sup.filter(UserWarning, message='Sample size too small')
        w, p = stats.wilcoxon(x, y, alternative='greater', correction=True, mode='approx')
    assert_equal(w, 27)
    assert_almost_equal(p, 0.3176447, decimal=6)