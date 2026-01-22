import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_qr_equiv(self):
    res2 = self.res1.model.fit(method='qr')
    assert_allclose(self.res1.HC0_se, res2.HC0_se)