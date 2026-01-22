import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_kurtosis_3d(self):
    x = np.tile(self.kurtosis_x, (10, 10, 1))
    kurtosis = np.array(robust_kurtosis(x, axis=2))
    for i, r in enumerate(self.expected_kurtosis):
        assert_almost_equal(r * np.ones((10, 10)), kurtosis[i])