import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_kurtosis_ab(self):
    x = self.kurtosis_x
    alpha, beta = (10.0, 45.0)
    kurtosis = robust_kurtosis(self.kurtosis_x, ab=(alpha, beta), excess=False)
    num = np.mean(x[x > np.percentile(x, 100.0 - alpha)]) - np.mean(x[x < np.percentile(x, alpha)])
    denom = np.mean(x[x > np.percentile(x, 100.0 - beta)]) - np.mean(x[x < np.percentile(x, beta)])
    assert_almost_equal(kurtosis[2], num / denom)