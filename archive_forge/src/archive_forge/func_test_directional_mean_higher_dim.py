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
def test_directional_mean_higher_dim(self):
    data = np.array([[0.8660254, 0.5, 0.0], [0.8660254, -0.5, 0.0]])
    full_array = np.tile(data, (2, 2, 2, 1))
    expected = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    dirstats = stats.directional_stats(full_array, axis=2)
    assert_allclose(expected, dirstats.mean_direction)