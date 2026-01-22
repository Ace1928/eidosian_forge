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
def test_formula_categorical(self):
    resp = self.resp
    data = ds.df
    formula = 'apply ~ pared + public + gpa - 1'
    modf2 = OrderedModel.from_formula(formula, data, distr='probit')
    resf2 = modf2.fit(method='bfgs', disp=False)
    assert_allclose(resf2.params, resp.params, atol=1e-08)
    assert modf2.exog_names == resp.model.exog_names
    assert modf2.data.ynames == resp.model.data.ynames
    assert hasattr(modf2.data, 'frame')
    assert not hasattr(modf2, 'frame')
    msg = 'Only ordered pandas Categorical'
    with pytest.raises(ValueError, match=msg):
        OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': np.asarray(data['apply']), 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr='probit')