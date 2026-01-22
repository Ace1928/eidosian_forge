import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_bcov_scaled(self):
    assert_almost_equal(self.res1.bcov_scaled, self.res2.h1, self.decimal_bcov_scaled)
    assert_almost_equal(self.res1.h2, self.res2.h2, self.decimal_bcov_scaled)
    assert_almost_equal(self.res1.h3, self.res2.h3, self.decimal_bcov_scaled)