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
def test_array_pandas(self):
    df = make_dataframe()
    df.iloc[[2, 5, 10], [2, 3, 1]] = np.nan
    y, X = (df[df.columns[0]].values, df[df.columns[1:]])
    data, _ = sm_data.handle_missing(y, X, missing='drop')
    df = df.dropna()
    y_exp, X_exp = (df[df.columns[0]].values, df[df.columns[1:]])
    assert_frame_equal(data['exog'], X_exp)
    np.testing.assert_array_equal(data['endog'], y_exp)