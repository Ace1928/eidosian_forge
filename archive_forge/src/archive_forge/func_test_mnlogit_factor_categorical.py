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
def test_mnlogit_factor_categorical():
    dta = sm.datasets.anes96.load_pandas()
    dta['endog'] = dta.endog.replace(dict(zip(range(7), 'ABCDEFG')))
    exog = sm.add_constant(dta.exog, prepend=True)
    mod = sm.MNLogit(dta.endog, exog)
    res = mod.fit(disp=0)
    dta['endog'] = dta['endog'].astype('category')
    mod = sm.MNLogit(dta.endog, exog)
    res_cat = mod.fit(disp=0)
    assert_allclose(res.params, res_cat.params)