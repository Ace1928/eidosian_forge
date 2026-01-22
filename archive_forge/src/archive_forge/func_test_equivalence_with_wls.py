import numpy as np
from numpy.testing import assert_allclose
import pytest
from statsmodels.regression.linear_model import WLS
from statsmodels.regression._tools import _MinimalWLS
@pytest.mark.parametrize('check', [True, False])
def test_equivalence_with_wls(self, check):
    res = WLS(self.endog1, self.exog1).fit()
    minres = _MinimalWLS(self.endog1, self.exog1, check_endog=check, check_weights=check).fit()
    assert_allclose(res.params, minres.params)
    assert_allclose(res.resid, minres.resid)
    res = WLS(self.endog2, self.exog2).fit()
    minres = _MinimalWLS(self.endog2, self.exog2, check_endog=check, check_weights=check).fit()
    assert_allclose(res.params, minres.params)
    assert_allclose(res.resid, minres.resid)
    res = WLS(self.endog1, self.exog1, weights=self.weights1).fit()
    minres = _MinimalWLS(self.endog1, self.exog1, weights=self.weights1, check_endog=check, check_weights=check).fit()
    assert_allclose(res.params, minres.params)
    assert_allclose(res.resid, minres.resid)
    res = WLS(self.endog2, self.exog2, weights=self.weights2).fit()
    minres = _MinimalWLS(self.endog2, self.exog2, weights=self.weights2, check_endog=check, check_weights=check).fit()
    assert_allclose(res.params, minres.params)
    assert_allclose(res.resid, minres.resid)