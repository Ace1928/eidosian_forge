from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests import test_dynamic_factor_mq_monte_carlo
@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('standardize', [(10, 10), ([10], [10]), (np.array(10), np.array(10)), (pd.Series([10], index=['y']), pd.Series([10], index=['y']))])
def test_simulate_standardized_1d(standardize, use_pandas):
    endog = np.zeros(100) + 10
    if use_pandas:
        endog = pd.Series(endog, name='y')
    mod = dynamic_factor_mq.DynamicFactorMQ(endog, factors=1, factor_orders=1, idiosyncratic_ar1=False, standardize=standardize)
    lambda1 = 2.0
    phi = 0.5
    params = [lambda1, phi, 0.0, 0]
    res = mod.smooth(params)
    mean = np.atleast_1d(standardize[0])[0]
    std = np.atleast_1d(standardize[1])[0]
    desired = phi ** np.arange(10) * lambda1 * std + mean
    desired_nd = desired[:, None] if use_pandas else desired[:, None, None]
    actual = res.simulate(10, initial_state=[1.0])
    assert_equal(actual.shape, (10,))
    assert_allclose(actual, desired)
    actual = res.simulate(10, initial_state=[1.0], repetitions=1)
    desired_shape = (10, 1) if use_pandas else (10, 1, 1)
    assert_equal(actual.shape, desired_shape)
    assert_allclose(actual, desired_nd)
    actual = res.simulate(10, initial_state=[1.0], repetitions=2)
    desired_shape = (10, 2) if use_pandas else (10, 1, 2)
    assert_equal(actual.shape, desired_shape)
    assert_allclose(actual, np.repeat(desired_nd, 2, axis=-1))