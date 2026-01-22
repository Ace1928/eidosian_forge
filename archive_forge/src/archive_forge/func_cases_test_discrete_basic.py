import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def cases_test_discrete_basic():
    seen = set()
    for distname, arg in distdiscrete:
        if distname in distslow:
            yield pytest.param(distname, arg, distname, marks=pytest.mark.slow)
        else:
            yield (distname, arg, distname not in seen)
        seen.add(distname)