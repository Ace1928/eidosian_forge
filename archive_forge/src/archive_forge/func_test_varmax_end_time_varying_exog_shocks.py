from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_varmax_end_time_varying_exog_shocks(reset_randomstate):
    endog = np.arange(1, 23).reshape(11, 2)
    exog = np.arange(1, 21) ** 2
    mod = varmax.VARMAX(endog[:10], trend='n', exog=exog[:10])
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
    mod2 = varmax.VARMAX(endog, trend='n', exog=exog[:11])
    res2 = mod2.filter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))
    tmp_exog = mod._validate_out_of_sample_exog(exog[10:], out_of_sample=10)
    with res._set_final_predicted_state(exog=tmp_exog, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    actual2 = res2.simulate(nsimulations, exog=exog[11:], anchor=-1, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=res2.predicted_state[:, -2])
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + exog[10 + i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)
    assert_allclose(actual2, desired)
    mod_actual = mod.simulate(res.params, nsimulations, exog=exog[10:], anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(mod_actual, desired)