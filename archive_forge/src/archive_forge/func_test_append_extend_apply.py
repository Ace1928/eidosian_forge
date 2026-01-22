from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
def test_append_extend_apply(reset_randomstate):
    endog, loadings, phi, sigma2, _, idio_var = gen_dfm_data(k_endog=10, nobs=100)
    endog1 = endog.iloc[:-10]
    endog2 = endog.iloc[-10:]
    mod = dynamic_factor_mq.DynamicFactorMQ(endog1, factor_orders=1, standardize=False, idiosyncratic_ar1=False)
    params = np.r_[loadings, phi, sigma2, idio_var]
    res = mod.smooth(params)
    msg = 'Cannot append data of a different dimension to a model.'
    with pytest.raises(ValueError, match=msg):
        res.append(endog2.iloc[:, :3])
    with pytest.raises(ValueError, match=msg):
        res.extend(endog2.iloc[:, :3])
    mod.initialize_known([0.1], [[1.0]])
    res2 = mod.smooth(params)
    assert_allclose(res.filter_results.initial_state, 0)
    assert_allclose(res.filter_results.initial_state_cov, 4 / 3.0)
    assert_allclose(res2.filter_results.initial_state, 0.1)
    assert_allclose(res2.filter_results.initial_state_cov, 1.0)
    res3 = res2.append(endog2, copy_initialization=False)
    assert_allclose(res3.filter_results.initial_state, 0)
    assert_allclose(res3.filter_results.initial_state_cov, 4 / 3.0)
    res4 = res2.append(endog2, copy_initialization=True)
    assert_allclose(res4.filter_results.initial_state, 0.1)
    assert_allclose(res4.filter_results.initial_state_cov, 1.0)
    res5 = res2.apply(endog, copy_initialization=False)
    assert_allclose(res5.filter_results.initial_state, 0)
    assert_allclose(res5.filter_results.initial_state_cov, 4 / 3.0)
    res6 = res2.apply(endog, copy_initialization=True)
    assert_allclose(res6.filter_results.initial_state, 0.1)
    assert_allclose(res6.filter_results.initial_state_cov, 1.0)