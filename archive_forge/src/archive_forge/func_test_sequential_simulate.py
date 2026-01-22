from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_sequential_simulate():
    n_simulations = 100
    mod = sarimax.SARIMAX([1], order=(0, 0, 0), trend='c')
    actual = mod.simulate([1, 0], n_simulations)
    assert_allclose(actual, np.ones(n_simulations))
    actual = mod.simulate([10, 0], n_simulations)
    assert_allclose(actual, np.ones(n_simulations) * 10)