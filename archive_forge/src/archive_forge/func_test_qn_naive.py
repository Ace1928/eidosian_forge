import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
def test_qn_naive(self):
    assert_almost_equal(scale.qn_scale(self.normal), scale._qn_naive(self.normal), DECIMAL)
    assert_almost_equal(scale.qn_scale(self.range), scale._qn_naive(self.range), DECIMAL)
    assert_almost_equal(scale.qn_scale(self.exponential), scale._qn_naive(self.exponential), DECIMAL)