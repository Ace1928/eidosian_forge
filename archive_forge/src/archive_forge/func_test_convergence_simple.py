from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_convergence_simple():
    gen = np.random.RandomState(0)
    e = gen.standard_normal(12000)
    y = e.copy()
    for i in range(1, e.shape[0]):
        y[i] = y[i - 1] - 0.2 * e[i - 1] + e[i]
    y = y[200:]
    mod = holtwinters.ExponentialSmoothing(y, initialization_method='estimated')
    res = mod.fit()
    ets_res = ETSModel(y).fit()
    assert_allclose(res.params['smoothing_level'], ets_res.smoothing_level, rtol=0.0001, atol=0.0001)
    assert_allclose(res.fittedvalues[10:], ets_res.fittedvalues[10:], rtol=0.0001, atol=0.0001)