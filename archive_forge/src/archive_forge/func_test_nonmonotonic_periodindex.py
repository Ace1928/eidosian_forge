from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_nonmonotonic_periodindex():
    tmp = pd.period_range(start=2000, end=2002, freq='Y')
    index = tmp.tolist() + tmp.tolist()
    endog = pd.Series(np.zeros(len(index)), index=index)
    message = 'A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.'
    with pytest.warns(ValueWarning, match=message):
        tsa_model.TimeSeriesModel(endog)