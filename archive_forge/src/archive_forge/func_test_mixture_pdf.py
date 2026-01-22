import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats
def test_mixture_pdf(self):
    mix = MixtureDistribution()
    grid = np.linspace(-4, 4, 10)
    res = mix.pdf(grid, [1 / 3.0, 2 / 3.0], dist=[stats.norm, stats.norm], kwargs=(dict(loc=-1, scale=0.25), dict(loc=1, scale=0.75)))
    npt.assert_almost_equal(res, np.array([7.92080017e-11, 1.05977272e-07, 3.823685e-05, 0.221485447, 0.100534607, 0.269531536, 0.321265627, 0.0939899015, 0.00674932493, 0.000118960201]))