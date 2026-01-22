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
def test_mnlogit_float_name():
    df = pd.DataFrame({'A': [0.0, 1.1, 0, 0, 1.1], 'B': [0, 1, 0, 1, 1]})
    with pytest.warns(SpecificationWarning, match='endog contains values are that not int-like'):
        result = smf.mnlogit(formula='A ~ B', data=df).fit()
    summ = result.summary().as_text()
    assert 'A=1.1' in summ