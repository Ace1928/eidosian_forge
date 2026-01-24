import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
def test_qn_robustbase(self):
    assert_almost_equal(scale.qn_scale(self.range), 13.3148, DECIMAL)
    assert_almost_equal(scale.qn_scale(self.stackloss), np.array([8.89286, 8.89286, 2.21914, 4.43828]), DECIMAL)
    assert_almost_equal(scale.qn_scale(self.sunspot[0:289]), 33.50901, DECIMAL)