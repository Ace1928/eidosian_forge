from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_pandas_anchor():
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.Series(np.zeros(2), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])
    desired = res.simulate(2, anchor=1, state_shocks=np.zeros(2), initial_state=np.zeros(1))
    actual = res.simulate(2, anchor=ix[1], state_shocks=np.zeros(2), initial_state=np.zeros(1))
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    actual = res.simulate(2, anchor=-1, state_shocks=np.zeros(2), initial_state=np.zeros(1))
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))