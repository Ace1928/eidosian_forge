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
def test_dist_keyword(self):
    x = stats.norm.rvs(size=20, random_state=12345)
    osm1, osr1 = stats.probplot(x, fit=False, dist='t', sparams=(3,))
    osm2, osr2 = stats.probplot(x, fit=False, dist=stats.t, sparams=(3,))
    assert_allclose(osm1, osm2)
    assert_allclose(osr1, osr2)
    assert_raises(ValueError, stats.probplot, x, dist='wrong-dist-name')
    assert_raises(AttributeError, stats.probplot, x, dist=[])

    class custom_dist:
        """Some class that looks just enough like a distribution."""

        def ppf(self, q):
            return stats.norm.ppf(q, loc=2)
    osm1, osr1 = stats.probplot(x, sparams=(2,), fit=False)
    osm2, osr2 = stats.probplot(x, dist=custom_dist(), fit=False)
    assert_allclose(osm1, osm2)
    assert_allclose(osr1, osr2)