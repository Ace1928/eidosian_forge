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
def test_mood_3d(self):
    shape = (10, 5, 6)
    np.random.seed(1234)
    x1 = np.random.randn(*shape)
    x2 = np.random.randn(*shape)
    for axis in range(3):
        z_vectest, pval_vectest = stats.mood(x1, x2, axis=axis)
        axes_idx = ([1, 2], [0, 2], [0, 1])
        for i in range(shape[axes_idx[axis][0]]):
            for j in range(shape[axes_idx[axis][1]]):
                if axis == 0:
                    slice1 = x1[:, i, j]
                    slice2 = x2[:, i, j]
                elif axis == 1:
                    slice1 = x1[i, :, j]
                    slice2 = x2[i, :, j]
                else:
                    slice1 = x1[i, j, :]
                    slice2 = x2[i, j, :]
                assert_array_almost_equal([z_vectest[i, j], pval_vectest[i, j]], stats.mood(slice1, slice2))