import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
@pytest.mark.parametrize('dist,args', distdiscrete)
def test_ppf_with_loc(dist, args):
    try:
        distfn = getattr(stats, dist)
    except TypeError:
        distfn = dist
    np.random.seed(1942349)
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        npt.assert_array_equal([_a - 1 + loc, _b + loc], [distfn.ppf(0.0, *args, loc=loc), distfn.ppf(1.0, *args, loc=loc)])