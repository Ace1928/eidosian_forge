from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_exact_distribution(self):
    p_tables = {3: self.pn3, 4: self.pn4, 5: self.pm5, 6: self.pm6}
    for n, table in p_tables.items():
        for m, p in table.items():
            u = np.arange(0, len(p))
            assert_allclose(_mwu_state.cdf(k=u, m=m, n=n), p, atol=0.001)
            u2 = np.arange(0, m * n + 1)
            assert_allclose(_mwu_state.cdf(k=u2, m=m, n=n) + _mwu_state.sf(k=u2, m=m, n=n) - _mwu_state.pmf(k=u2, m=m, n=n), 1)
            pmf = _mwu_state.pmf(k=u2, m=m, n=n)
            assert_allclose(pmf, pmf[::-1])
            pmf2 = _mwu_state.pmf(k=u2, m=n, n=m)
            assert_allclose(pmf, pmf2)