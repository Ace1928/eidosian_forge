import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
def test_arma_kwargs():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    _, res1_imle = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1)
    assert_equal(res1_imle.arma_estimator_kwargs, {})
    assert_equal(res1_imle.arma_results[1].minimize_results.message, 'Optimization terminated successfully.')
    arma_estimator_kwargs = {'minimize_kwargs': {'method': 'L-BFGS-B'}}
    _, res2_imle = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1, arma_estimator_kwargs=arma_estimator_kwargs)
    assert_equal(res2_imle.arma_estimator_kwargs, arma_estimator_kwargs)
    msg = res2_imle.arma_results[1].minimize_results.message
    if isinstance(msg, bytes):
        msg = msg.decode('utf-8')
    assert_equal(msg, 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH')