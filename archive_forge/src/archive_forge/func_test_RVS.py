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
@pytest.mark.parametrize('rng', rngs)
@pytest.mark.parametrize('size_in, size_out', sizes)
def test_RVS(self, rng, size_in, size_out):
    dist = StandardNormal()
    fni = NumericalInverseHermite(dist)
    rng2 = deepcopy(rng)
    rvs = fni.rvs(size=size_in, random_state=rng)
    if size_in is not None:
        assert rvs.shape == size_out
    if rng2 is not None:
        rng2 = check_random_state(rng2)
        uniform = rng2.uniform(size=size_in)
        rvs2 = stats.norm.ppf(uniform)
        assert_allclose(rvs, rvs2)