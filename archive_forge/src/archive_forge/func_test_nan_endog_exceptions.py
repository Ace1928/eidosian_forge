import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
def test_nan_endog_exceptions():
    nobs = 15
    y = np.repeat(np.arange(3), nobs // 3)
    x = np.column_stack((np.ones(nobs), np.arange(nobs)))
    with pytest.raises(ValueError, match='not be a constant'):
        OrderedModel(y, x, distr='logit')
    y_nan = y.astype(float)
    y_nan[0] = np.nan
    with pytest.raises(ValueError, match='NaN in dependent variable'):
        OrderedModel(y_nan, x[:, 1:], distr='logit')
    if hasattr(pd, 'CategoricalDtype'):
        df = pd.DataFrame({'endog': pd.Series(y, dtype=pd.CategoricalDtype([1, 2, 3], ordered=True)), 'exog': x[:, 1]})
        msg = 'missing values in categorical endog'
        with pytest.raises(ValueError, match=msg):
            OrderedModel(df['endog'], df[['exog']])