from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
def test_formula_missing_extra_arrays():
    np.random.seed(1)
    y = np.random.randn(10)
    y_missing = y.copy()
    y_missing[[2, 5]] = np.nan
    X = np.random.randn(10)
    X_missing = X.copy()
    X_missing[[1, 3]] = np.nan
    weights = np.random.uniform(size=10)
    weights_missing = weights.copy()
    weights_missing[[6]] = np.nan
    weights_wrong_size = np.random.randn(12)
    data = {'y': y, 'X': X, 'y_missing': y_missing, 'X_missing': X_missing, 'weights': weights, 'weights_missing': weights_missing}
    data = pd.DataFrame.from_dict(data)
    data['constant'] = 1
    formula = 'y_missing ~ X_missing'
    (endog, exog), missing_idx, design_info = handle_formula_data(data, None, formula, depth=2, missing='drop')
    kwargs = {'missing_idx': missing_idx, 'missing': 'drop', 'weights': data['weights_missing']}
    model_data = sm_data.handle_data(endog, exog, **kwargs)
    data_nona = data.dropna()
    assert_equal(data_nona['y'].values, model_data.endog)
    assert_equal(data_nona[['constant', 'X']].values, model_data.exog)
    assert_equal(data_nona['weights'].values, model_data.weights)
    tmp = handle_formula_data(data, None, formula, depth=2, missing='drop')
    (endog, exog), missing_idx, design_info = tmp
    weights_2d = np.random.randn(10, 10)
    weights_2d[[8, 7], [7, 8]] = np.nan
    kwargs.update({'weights': weights_2d, 'missing_idx': missing_idx})
    model_data2 = sm_data.handle_data(endog, exog, **kwargs)
    good_idx = [0, 4, 6, 9]
    assert_equal(data.loc[good_idx, 'y'], model_data2.endog)
    assert_equal(data.loc[good_idx, ['constant', 'X']], model_data2.exog)
    assert_equal(weights_2d[good_idx][:, good_idx], model_data2.weights)
    tmp = handle_formula_data(data, None, formula, depth=2, missing='drop')
    (endog, exog), missing_idx, design_info = tmp
    kwargs.update({'weights': weights_wrong_size, 'missing_idx': missing_idx})
    assert_raises(ValueError, sm_data.handle_data, endog, exog, **kwargs)