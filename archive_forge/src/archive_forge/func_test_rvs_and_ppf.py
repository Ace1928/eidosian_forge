import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
@pytest.mark.parametrize('distname, args', dists_with_params)
def test_rvs_and_ppf(distname, args):
    urng = np.random.default_rng(9807324628097097)
    rng1 = getattr(stats, distname)(*args)
    rvs1 = rng1.rvs(size=500, random_state=urng)
    rng2 = FastGeneratorInversion(rng1, random_state=urng)
    rvs2 = rng2.rvs(size=500)
    assert stats.cramervonmises_2samp(rvs1, rvs2).pvalue > 0.01
    q = [0.001, 0.1, 0.5, 0.9, 0.999]
    assert_allclose(rng1.ppf(q), rng2.ppf(q), atol=1e-10)