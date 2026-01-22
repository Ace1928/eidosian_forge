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
def test_example2b(self):
    t1 = [194, 15, 41, 29, 33, 181]
    t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
    t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
    t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29, 118, 25, 156, 310, 76, 26, 44, 23, 62]
    t5 = [130, 208, 70, 101, 208]
    t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
    t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
    t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5, 12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
    t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82, 54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
    t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36, 22, 139, 210, 97, 30, 23, 13, 14]
    t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
    t12 = [50, 254, 5, 283, 35, 12]
    t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
    t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66, 61, 34]
    Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14), midrank=True)
    assert_almost_equal(Tk, 3.294, 3)
    assert_array_almost_equal([0.599, 1.3269, 1.8052, 2.2486, 2.8009], tm[0:5], 4)
    assert_allclose(p, 0.0041, atol=0.00025)