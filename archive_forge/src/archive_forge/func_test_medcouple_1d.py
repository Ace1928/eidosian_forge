import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_medcouple_1d(self):
    x = np.reshape(np.arange(100.0), (50, 2))
    assert_raises(ValueError, _medcouple_1d, x)