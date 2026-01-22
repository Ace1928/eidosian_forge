import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_kurtosis_1d_2d(self, reset_randomstate):
    x = np.random.randn(100)
    y = x[:, None]
    kr_x = np.array(robust_kurtosis(x))
    kr_y = np.array(robust_kurtosis(y, axis=None))
    assert_almost_equal(kr_x, kr_y)