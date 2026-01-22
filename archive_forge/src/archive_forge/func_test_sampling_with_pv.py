import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
@pytest.mark.parametrize('pv', [[0.18, 0.02, 0.8], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
def test_sampling_with_pv(self, pv):
    pv = np.asarray(pv, dtype=np.float64)
    rng = DiscreteAliasUrn(pv, random_state=123)
    rng.rvs(100000)
    pv = pv / pv.sum()
    variates = np.arange(0, len(pv))
    m_expected = np.average(variates, weights=pv)
    v_expected = np.average((variates - m_expected) ** 2, weights=pv)
    mv_expected = (m_expected, v_expected)
    check_discr_samples(rng, pv, mv_expected)