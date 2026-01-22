import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats
def test_mixture_rvs_fixed(self):
    mix = MixtureDistribution()
    np.random.seed(1234)
    res = mix.rvs([0.15, 0.85], 50, dist=[stats.norm, stats.norm], kwargs=(dict(loc=1, scale=0.5), dict(loc=-1, scale=0.5)))
    npt.assert_almost_equal(res, np.array([-0.5794956, -1.72290504, -1.70098664, -1.0504591, -1.27412122, -1.07230975, -0.82298983, -1.01775651, -0.71713085, -0.2271706, -1.48711817, -1.03517244, -0.84601557, -1.10424938, -0.48309963, -2.20022682, 0.01530181, 1.1238961, -1.57131564, -0.89405831, -0.64763969, -1.39271761, 0.55142161, -0.76897013, -0.64788589, -0.73824602, -1.46312716, 0.00392148, -0.88651873, -1.57632955, -0.68401028, -0.98024366, -0.76780384, 0.93160258, -2.78175833, -0.33944719, -0.92368472, -0.91773523, -1.21504785, -0.61631563, 1.0091446, -0.50754008, 1.37770699, -0.86458208, -0.3040069, -0.96007884, 1.10763429, -1.19998229, -1.51392528, -1.29235911]))