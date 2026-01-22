import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson
@pytest.mark.xfail(reason='Different computation of variances')
def test_nonstationary_series_variance():
    endog = np.arange(1, 12) * 1.0
    res, _ = durbin_levinson(endog, 2, demean=False)
    desired_sigma2 = 15.36526603
    assert_allclose(res[2].sigma2, desired_sigma2)