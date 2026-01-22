import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_wallenius_against_mpmath(self):
    M = 50
    n = 30
    N = 20
    odds = 2.25
    sup = np.arange(21)
    pmf = np.array([3.699003068656875e-20, 5.89398584245431e-17, 2.1594437742911123e-14, 3.221458044649955e-12, 2.4658279241205077e-10, 1.0965862603981212e-08, 3.057890479665704e-07, 5.622818831643761e-06, 7.056482841531681e-05, 0.000618899425358671, 0.003854172932571669, 0.01720592676256026, 0.05528844897093792, 0.12772363313574242, 0.21065898367825722, 0.24465958845359234, 0.1955114898110033, 0.10355390084949237, 0.03414490375225675, 0.006231989845775931, 0.0004715577304677075])
    mean = 14.808018384813426
    var = 2.6085975877923717
    assert_allclose(nchypergeom_wallenius.pmf(sup, M, n, N, odds), pmf, rtol=1e-13, atol=1e-13)
    assert_allclose(nchypergeom_wallenius.mean(M, n, N, odds), mean, rtol=1e-13)
    assert_allclose(nchypergeom_wallenius.var(M, n, N, odds), var, rtol=1e-11)