import pickle
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
import pytest
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .results import results_kalman_filter
def test_unobserved_components_pickle():
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog2 = np.random.normal(size=(nobs, 2))
    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)
    models = [UnobservedComponents(endog, 'llevel', exog=exog2), UnobservedComponents(endog_pd, 'llevel', exog=exog2_pd)]
    for mod in models:
        pkl_mod = pickle.loads(pickle.dumps(mod))
        assert_equal(mod.start_params, pkl_mod.start_params)
        res = mod.fit(disp=False)
        pkl_res = pkl_mod.fit(disp=False)
        assert_allclose(res.llf_obs, pkl_res.llf_obs)
        assert_allclose(res.tvalues, pkl_res.tvalues)
        assert_allclose(res.smoothed_state, pkl_res.smoothed_state)
        assert_allclose(res.resid, pkl_res.resid)
        assert_allclose(res.impulse_responses(10), res.impulse_responses(10))