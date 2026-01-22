import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats
def test_mv_mixture_rvs_fixed(self):
    np.random.seed(1234)
    cov3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])
    mu = np.array([-1, 0.0, 2.0])
    mu2 = np.array([4, 2.0, 2.0])
    mvn3 = mvd.MVNormal(mu, cov3)
    mvn32 = mvd.MVNormal(mu2, cov3 / 2)
    res = mv_mixture_rvs([0.2, 0.8], 10, [mvn3, mvn32], 3)
    npt.assert_almost_equal(res, np.array([[-0.23955497, 1.73426482, 0.36100243], [2.52063189, 1.0832677, 1.89947131], [4.36755379, 2.14480498, 2.22003966], [3.1141545, 1.21250505, 2.58511199], [4.1980202, 2.50017561, 1.87324933], [3.48717503, 0.91847424, 2.14004598], [3.55904133, 2.74367622, 0.68619582], [3.60521933, 1.57316531, 0.82784584], [3.86102275, 0.6211812, 1.33016426], [3.91074761, 2.037155, 2.22247051]]))