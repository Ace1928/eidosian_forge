import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_bcov_unscaled(self):
    if not hasattr(self.res2, 'bcov_unscaled'):
        pytest.skip('No unscaled cov matrix from SAS')
    assert_almost_equal(self.res1.bcov_unscaled, self.res2.bcov_unscaled, DECIMAL_4)