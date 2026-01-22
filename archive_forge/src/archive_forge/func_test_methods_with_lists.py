import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
@pytest.mark.parametrize('method', ['pdf', 'logpdf', 'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', distcont)
def test_methods_with_lists(method, distname, args):
    dist = getattr(stats, distname)
    f = getattr(dist, method)
    if distname == 'invweibull' and method.startswith('log'):
        x = [1.5, 2]
    else:
        x = [0.1, 0.2]
    shape2 = [[a] * 2 for a in args]
    loc = [0, 0.1]
    scale = [1, 1.01]
    result = f(x, *shape2, loc=loc, scale=scale)
    npt.assert_allclose(result, [f(*v) for v in zip(x, *shape2, loc, scale)], rtol=1e-14, atol=5e-14)