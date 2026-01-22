from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_states_index_int64index():
    nobs = 10
    ix = pd.Index(np.arange(10))
    endog = pd.Series(np.zeros(nobs), index=ix)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])
    predicted_ix = pd.Index(np.arange(11))
    cols = pd.Index(['state.0', 'state.1'])
    check_states_index(res.states, ix, predicted_ix, cols)