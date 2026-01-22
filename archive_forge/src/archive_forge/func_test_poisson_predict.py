from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_poisson_predict():
    data = load_randhie()
    exog = sm.add_constant(data.exog, prepend=True)
    res = sm.Poisson(data.endog, exog).fit(method='newton', disp=0)
    pred1 = res.predict()
    pred2 = res.predict(exog)
    assert_almost_equal(pred1, pred2)
    pred3 = res.predict(exog, offset=0, exposure=1)
    assert_almost_equal(pred1, pred3)
    pred3 = res.predict(exog, offset=0, exposure=2)
    assert_almost_equal(2 * pred1, pred3)
    pred3 = res.predict(exog, offset=np.log(2), exposure=1)
    assert_almost_equal(2 * pred1, pred3)