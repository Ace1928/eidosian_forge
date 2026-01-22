import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import dowj, lake
from statsmodels.tsa.arima.estimators.durbin_levinson import durbin_levinson
def test_nonstationary_series():
    endog = np.arange(1, 12) * 1.0
    res, _ = durbin_levinson(endog, 2, demean=False)
    desired_ar_params = [0.92318534179, -0.06166314306]
    assert_allclose(res[2].ar_params, desired_ar_params)